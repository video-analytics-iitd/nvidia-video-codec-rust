use bincode::{Decode, Encode};
use core::panic;

use std::{
    collections::VecDeque,
    ffi::{c_uchar, c_void},
    mem::MaybeUninit,
    ptr,
    sync::{Arc, Mutex},
};

use cudarc::driver::{
    sys::{cuMemAlloc_v2, cuMemcpy2DAsync_v2, CUdeviceptr, CUmemorytype, CUresult, CUDA_MEMCPY2D},
    CudaContext, CudaSlice, CudaStream, DevicePtr,
};

use crate::sys::{
    cuviddec::{
        cudaVideoChromaFormat_enum, cudaVideoCodec_enum, cudaVideoCreateFlags_enum,
        cudaVideoDeinterlaceMode_enum, cudaVideoSurfaceFormat_enum, cuvidDecodeStatus_enum,
        CUvideoctxlock, CUvideodecoder, CUVIDDECODECAPS, CUVIDDECODECREATEINFO,
        CUVIDGETDECODESTATUS, CUVIDPICPARAMS, CUVIDPROCPARAMS,
    },
    nvcuvid::{
        CUvideopacketflags, CUVIDEOFORMAT, CUVIDEOFORMATEX, CUVIDOPERATINGPOINTINFO,
        CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDSOURCEDATAPACKET,
    },
};

use super::{result::DecodeError, DECODE_API};

// type Device = Arc<CudaDevice>;
type Ctx = Arc<CudaContext>;
type Stream = Arc<CudaStream>;
///
#[derive(Debug)]
struct ParserPtr(*mut c_void);
unsafe impl Send for ParserPtr {}

#[derive(Debug)]
struct DecCtxPtr(*mut c_void);
unsafe impl Send for DecCtxPtr {}

///
#[derive(Debug)]
pub struct Decoder {
    parser: ParserPtr,
    // Used to make sure that CudaDevice stays alive while the Decoder does
    _ctx: Ctx,
    _stream: Stream,
    // Context for the callbacks
    decoder_context: DecCtxPtr,
}

impl Drop for Decoder {
    fn drop(&mut self) {}
}

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct Rect {
    l: i32,
    t: i32,
    r: i32,
    b: i32,
}

#[derive(Encode, Decode, PartialEq, Debug, Clone, Copy)]
pub struct Dim {
    pub w: i32,
    pub h: i32,
}

const MAX_FRM_CNT: usize = 32;
pub struct DecodeContext {
    frame_queue: Mutex<VecDeque<Arc<CudaSlice<f32>>>>,
    // cuda_device: Arc<CudaDevice>,
    ctx: Arc<CudaContext>,
    cuda_stream: Arc<CudaStream>,
    codec: cudaVideoCodec_enum,
    // e_chroma_format: cudaVideoChromaFormat_enum,
    n_bit_depth_minus8: u8,
    n_bpp: usize,
    chroma_format: cudaVideoChromaFormat_enum,
    output_format: cudaVideoSurfaceFormat_enum,
    video_format: CUVIDEOFORMAT,
    num_dec_surfaces: i32,
    ctx_lock: CUvideoctxlock,
    max_width: u32,
    max_height: u32,
    crop_rect: Rect,
    resize_dim: Dim,
    width: u32,
    luma_height: u32,
    chroma_height: u32,
    num_chroma_planes: u32,

    display_rect: Rect,
    surface_height: u32,
    surface_width: u32,
    decoder: CUvideodecoder,

    decode_pic_cnt: i32,
    pic_number_in_decode_order: [i32; MAX_FRM_CNT],

    operating_pts: i32,
    display_all_layers: bool,
}

impl DecodeContext {
    pub fn new(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        codec_type: cudaVideoCodec_enum,
        resize_info: Dim,
    ) -> Self {
        let mut ctx_lock = ptr::null_mut();
        unsafe { (DECODE_API.ctx_lock_create)(&mut ctx_lock, ctx.cu_ctx()) };
        DecodeContext {
            frame_queue: Mutex::new(VecDeque::new()),
            // cuda_device: cuda_dev,
            codec: codec_type,
            n_bit_depth_minus8: 0,
            n_bpp: 1,
            chroma_format: cudaVideoChromaFormat_enum::cudaVideoChromaFormat_420,
            output_format: cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV12,
            video_format: CUVIDEOFORMAT::default(),
            num_dec_surfaces: 0,
            ctx_lock,
            max_width: 0,
            max_height: 0,
            crop_rect: Rect {
                l: 0,
                r: 0,
                t: 0,
                b: 0,
            },
            resize_dim: resize_info,
            // resize_dim: Dim { w: 0, h: 0 },
            width: 0,
            luma_height: 0,
            chroma_height: 0,
            num_chroma_planes: 0,
            display_rect: Rect {
                l: 0,
                r: 0,
                t: 0,
                b: 0,
            },
            surface_height: 0,
            surface_width: 0,
            decoder: ptr::null_mut(),
            decode_pic_cnt: 0,
            pic_number_in_decode_order: [0; MAX_FRM_CNT],
            operating_pts: 0,
            display_all_layers: false,
            ctx,
            cuda_stream: stream,
        }
    }
}

fn get_chroma_height_factor(surface_format: cudaVideoSurfaceFormat_enum) -> f32 {
    match surface_format {
        cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV12
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P016 => 0.5,

        cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444_16Bit
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV16
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P216 => 1.0,
    }
}

fn get_chroma_plane_count(surface_format: cudaVideoSurfaceFormat_enum) -> i32 {
    match surface_format {
        cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV12
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P016
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV16
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P216 => 1,

        cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444_16Bit => 2,
    }
}

unsafe extern "C" fn handle_video_seq(
    arg1: *mut ::core::ffi::c_void,
    p_video_format: *mut CUVIDEOFORMAT,
) -> ::core::ffi::c_int {
    println!("Inside the callback: [Handle Video Sequence]");
    println!("Fields of video format: {:?}", *p_video_format);
    let context = arg1 as *mut DecodeContext;
    unsafe {
        let cu_ctx = (*context).ctx.clone(); // Replace with your actual field name
        let status = cudarc::driver::sys::cuCtxPushCurrent_v2((*cu_ctx).cu_ctx());
        if status != CUresult::CUDA_SUCCESS {
            panic!("Failed to push CUDA context: {:?}", status);
        }
    }
    let n_decode_surface = (*p_video_format).min_num_decode_surfaces as i32;
    let mut decode_caps: CUVIDDECODECAPS = unsafe { std::mem::zeroed() };
    decode_caps.eCodecType = (*p_video_format).codec;
    decode_caps.eChromaFormat = (*p_video_format).chroma_format;
    decode_caps.nBitDepthMinus8 = (*p_video_format).bit_depth_luma_minus8 as u32;

    let res = unsafe { (DECODE_API.get_decode_caps)(&mut decode_caps) };
    println!("{:?}", res);

    if decode_caps.bIsSupported == 0 {
        panic!("Codec not supported on this GPU")
    }

    if decode_caps.nMaxWidth < (*p_video_format).coded_width
        || decode_caps.nMaxHeight < (*p_video_format).coded_height
    {
        panic!(
            "Decoder capability is {}x{}. Video size is {}x{}",
            decode_caps.nMaxWidth,
            decode_caps.nMaxHeight,
            (*p_video_format).coded_width,
            (*p_video_format).coded_height
        );
    }

    let width_in_mb = (*p_video_format).coded_width >> 4;
    let height_in_mb = (*p_video_format).coded_height >> 4;
    let total_mb = width_in_mb * height_in_mb;

    if total_mb > decode_caps.nMaxMBCount {
        panic!(
            "Decoder max macroblock count is {}. Video has {} macroblocks ({}x{}).",
            decode_caps.nMaxMBCount, total_mb, width_in_mb, height_in_mb
        );
    }
    (*context).codec = (*p_video_format).codec;
    (*context).chroma_format = (*p_video_format).chroma_format;
    (*context).n_bit_depth_minus8 = (*p_video_format).bit_depth_luma_minus8;
    if (*context).n_bit_depth_minus8 > 0 {
        (*context).n_bpp = 2;
    } else {
        (*context).n_bpp = 1;
    }

    (*context).output_format = match (*context).chroma_format {
        cudaVideoChromaFormat_enum::cudaVideoChromaFormat_420
        | cudaVideoChromaFormat_enum::cudaVideoChromaFormat_Monochrome => {
            if (*p_video_format).bit_depth_luma_minus8 != 0 {
                cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P016
            } else {
                cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV12
            }
        }
        cudaVideoChromaFormat_enum::cudaVideoChromaFormat_444 => {
            if (*p_video_format).bit_depth_luma_minus8 != 0 {
                cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444_16Bit
            } else {
                cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444
            }
        }
        cudaVideoChromaFormat_enum::cudaVideoChromaFormat_422 => {
            if (*p_video_format).bit_depth_luma_minus8 != 0 {
                cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P216
            } else {
                cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV16
            }
        }
    };

    if decode_caps.nOutputFormatMask & (1 << ((*context).output_format as i32)) == 0 {
        if decode_caps.nOutputFormatMask
            & (1 << (cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV12 as i32))
            != 0
        {
            (*context).output_format = cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV12;
        } else if decode_caps.nOutputFormatMask
            & (1 << (cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P016 as i32))
            != 0
        {
            (*context).output_format = cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P016;
        } else if decode_caps.nOutputFormatMask
            & (1 << (cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444 as i32))
            != 0
        {
            (*context).output_format = cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444;
        } else if decode_caps.nOutputFormatMask
            & (1 << (cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444_16Bit as i32))
            != 0
        {
            (*context).output_format =
                cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_YUV444_16Bit;
        } else if decode_caps.nOutputFormatMask
            & (1 << (cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV16 as i32))
            != 0
        {
            (*context).output_format = cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV16;
        } else if decode_caps.nOutputFormatMask
            & (1 << (cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P216 as i32))
            != 0
        {
            (*context).output_format = cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P216;
        } else {
            panic!("No supported output format found: CUDA_ERROR_NOT_SUPPORTED");
        }
    }

    (*context).video_format = (*p_video_format).clone();
    let mut video_decode_create_info =
        create_viddec_info(p_video_format, &mut (*context), n_decode_surface);

    (*context).chroma_height = ((*context).luma_height as f32
        * get_chroma_height_factor((*context).output_format))
    .ceil() as u32;
    (*context).num_chroma_planes = get_chroma_plane_count((*context).output_format) as u32;

    (*context).surface_height = video_decode_create_info.ulTargetHeight as u32;
    (*context).surface_width = video_decode_create_info.ulTargetWidth as u32;

    (*context).display_rect.b = video_decode_create_info.display_area.bottom as i32;
    (*context).display_rect.t = video_decode_create_info.display_area.top as i32;
    (*context).display_rect.l = video_decode_create_info.display_area.left as i32;
    (*context).display_rect.r = video_decode_create_info.display_area.right as i32;

    let res = unsafe {
        (DECODE_API.create_decoder)(&mut (*context).decoder, &mut video_decode_create_info)
    };
    if res != CUresult::CUDA_SUCCESS {
        panic!("Something went wrong during decoder init");
    }

    n_decode_surface
}

fn create_viddec_info(
    p_video_format: *mut CUVIDEOFORMAT,
    context_box: &mut DecodeContext,
    n_decode_surface: i32,
) -> crate::sys::cuviddec::CUVIDDECODECREATEINFO {
    let mut video_decode_create_info = MaybeUninit::<CUVIDDECODECREATEINFO>::uninit();
    unsafe {
        ptr::write_bytes(
            video_decode_create_info.as_mut_ptr() as *mut u8,
            0,
            std::mem::size_of::<CUVIDDECODECREATEINFO>(),
        );
    }
    unsafe {
        let ptr: *mut crate::sys::cuviddec::_CUVIDDECODECREATEINFO =
            video_decode_create_info.as_mut_ptr();
        (*ptr).CodecType = (*p_video_format).codec;
        (*ptr).ChromaFormat = (*p_video_format).chroma_format;
        (*ptr).OutputFormat = context_box.output_format;
        (*ptr).bitDepthMinus8 = (*p_video_format).bit_depth_luma_minus8 as u64;
        if (*p_video_format).progressive_sequence != 0 {
            (*ptr).DeinterlaceMode = cudaVideoDeinterlaceMode_enum::cudaVideoDeinterlaceMode_Weave;
        } else {
            (*ptr).DeinterlaceMode =
                cudaVideoDeinterlaceMode_enum::cudaVideoDeinterlaceMode_Adaptive;
        }
        (*ptr).ulNumOutputSurfaces = 2;
        // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
        (*ptr).ulCreationFlags = cudaVideoCreateFlags_enum::cudaVideoCreate_PreferCUVID as u64;
        if context_box.num_dec_surfaces == 0 || context_box.num_dec_surfaces > n_decode_surface {
            context_box.num_dec_surfaces = n_decode_surface;
        }
        (*ptr).ulNumDecodeSurfaces = context_box.num_dec_surfaces as u64;
        (*ptr).vidLock = context_box.ctx_lock;
        (*ptr).ulWidth = (*p_video_format).coded_width as u64;
        (*ptr).ulHeight = (*p_video_format).coded_height as u64;

        if (*p_video_format).codec == cudaVideoCodec_enum::cudaVideoCodec_AV1
            && (*p_video_format).seqhdr_data_length > 0
        {
            // Don't overwrite if already set from command line or config
            if !(context_box.max_width > (*p_video_format).coded_width as u32
                || context_box.max_height > (*p_video_format).coded_height as u32)
            {
                // let vid_format_ex =
                //     unsafe { &*((*p_video_format) as *const CUVIDEOFORMAT as *const CUVIDEOFORMATEX) };
                let vid_format_ex = &*(p_video_format as *const CUVIDEOFORMATEX);
                context_box.max_width = vid_format_ex.__bindgen_anon_1.av1.max_width as u32;
                context_box.max_height = vid_format_ex.__bindgen_anon_1.av1.max_height as u32;
            }
        }

        if context_box.max_width < (*p_video_format).coded_width as u32 {
            context_box.max_width = (*p_video_format).coded_width as u32;
        }
        if context_box.max_height < (*p_video_format).coded_height as u32 {
            context_box.max_height = (*p_video_format).coded_height as u32;
        }

        (*ptr).ulMaxWidth = context_box.max_width as u64;
        (*ptr).ulMaxHeight = context_box.max_height as u64;

        if (context_box.crop_rect.r == 0 && context_box.crop_rect.b == 0)
            && (context_box.resize_dim.w == 0 && context_box.resize_dim.h == 0)
        {
            context_box.width =
                ((*p_video_format).display_area.right - (*p_video_format).display_area.left) as u32;
            context_box.luma_height =
                ((*p_video_format).display_area.bottom - (*p_video_format).display_area.top) as u32;

            (*ptr).ulTargetWidth = (*p_video_format).coded_width as u64;
            (*ptr).ulTargetHeight = (*p_video_format).coded_height as u64;
        } else {
            if context_box.resize_dim.w > 0 && context_box.resize_dim.h > 0 {
                (*ptr).display_area.left = (*p_video_format).display_area.left as i16;
                (*ptr).display_area.top = (*p_video_format).display_area.top as i16;
                (*ptr).display_area.right = (*p_video_format).display_area.right as i16;

                (*ptr).display_area.bottom = (*p_video_format).display_area.bottom as i16;

                context_box.width = context_box.resize_dim.w as u32;
                context_box.luma_height = context_box.resize_dim.h as u32;
            }

            if context_box.crop_rect.r > 0 && context_box.crop_rect.b > 0 {
                (*ptr).display_area.left = context_box.crop_rect.l as i16;
                (*ptr).display_area.top = context_box.crop_rect.t as i16;
                (*ptr).display_area.right = context_box.crop_rect.r as i16;
                (*ptr).display_area.bottom = context_box.crop_rect.b as i16;

                context_box.width = (context_box.crop_rect.r - context_box.crop_rect.l) as u32;
                context_box.luma_height =
                    (context_box.crop_rect.b - context_box.crop_rect.t) as u32;
            }

            (*ptr).ulTargetWidth = context_box.width as u64;
            (*ptr).ulTargetHeight = context_box.luma_height as u64;
        }
        video_decode_create_info.assume_init()
    }
}

unsafe extern "C" fn handle_picture_decode(
    arg1: *mut ::core::ffi::c_void,
    pic_params: *mut CUVIDPICPARAMS,
) -> ::core::ffi::c_int {
    // println!("Inside the callback: [Handle Picture Decode]");
    let context_box = arg1 as *mut DecodeContext;

    if (*context_box).decoder == ptr::null_mut() {
        panic!("Decoder not initialized.")
    }

    (*context_box).pic_number_in_decode_order[(*pic_params).CurrPicIdx as usize] =
        (*context_box).decode_pic_cnt;

    (*context_box).decode_pic_cnt += 1;

    unsafe { (DECODE_API.decode_picture)((*context_box).decoder, pic_params) };

    1
}

unsafe extern "C" fn handle_picture_display(
    arg1: *mut ::core::ffi::c_void,
    p_disp_info: *mut CUVIDPARSERDISPINFO,
) -> ::core::ffi::c_int {
    let context = arg1 as *mut DecodeContext;
    let mut video_processing_parameters = CUVIDPROCPARAMS {
        progressive_frame: (*p_disp_info).progressive_frame,
        second_field: (*p_disp_info).repeat_first_field + 1,
        top_field_first: (*p_disp_info).top_field_first,
        unpaired_field: ((*p_disp_info).repeat_first_field < 0) as i32,
        output_stream: (*context).cuda_stream.cu_stream().clone(),
        ..Default::default()
    };

    let mut dp_src_frame: CUdeviceptr = 0;
    let mut src_pitch: u32 = 0;

    unsafe {
        (DECODE_API.map_video_frame)(
            (*context).decoder,
            (*p_disp_info).picture_index,
            &mut dp_src_frame,
            &mut src_pitch,
            &mut video_processing_parameters,
        )
    };

    let mut decode_status: CUVIDGETDECODESTATUS = unsafe { std::mem::zeroed() };
    unsafe {
        let result: cudarc::driver::sys::CUresult = (DECODE_API.get_decode_status)(
            (*context).decoder,
            (*p_disp_info).picture_index,
            &mut decode_status,
        );
        if result == CUresult::CUDA_SUCCESS
            && (decode_status.decodeStatus == cuvidDecodeStatus_enum::cuvidDecodeStatus_Error
                || decode_status.decodeStatus
                    == cuvidDecodeStatus_enum::cuvidDecodeStatus_Error_Concealed)
        {
            println!(
                "Decode error occured for the picture {}",
                (*context).pic_number_in_decode_order[(*p_disp_info).picture_index as usize]
            )
        }
    };

    let mut frame: CUdeviceptr = 0;
    assert!((*context).width != 0, "width must not be zero");
    let temp_width = match (*context).output_format {
        cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV12
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P016
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_NV16
        | cudaVideoSurfaceFormat_enum::cudaVideoSurfaceFormat_P216 => ((*context).width + 1) & !1,
        _ => (*context).width,
    };
    let frame_size: i32 = temp_width as i32
        * ((*context).luma_height + ((*context).chroma_height * (*context).num_chroma_planes))
            as i32
        * (*context).n_bpp as i32;

    unsafe { cuMemAlloc_v2(&mut frame, frame_size as usize) };

    // now copy the luma and chroma planes
    let mut m = ::core::mem::MaybeUninit::<CUDA_MEMCPY2D>::uninit();
    let mut m = unsafe {
        ::core::ptr::write_bytes(m.as_mut_ptr(), 0, 1);
        m.assume_init()
    };
    m.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    m.srcDevice = dp_src_frame;
    m.srcPitch = src_pitch as usize;
    m.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    m.dstDevice = frame;
    m.dstPitch = (*context).n_bpp * temp_width as usize;
    m.WidthInBytes = (*context).n_bpp * temp_width as usize;
    m.Height = (*context).luma_height as usize;

    cuMemcpy2DAsync_v2(&m, &mut *(*context).cuda_stream.cu_stream());

    let offset = m.srcPitch * (((*context).surface_height as usize + 1) & !1);
    let ptr = (dp_src_frame as *mut u8).wrapping_add(offset);
    m.srcDevice = ptr as CUdeviceptr;

    // m.dstHost = latest_frame as usize + m.dstPitch as usize + (*context_box).luma_height as usize;
    let offset = m.dstPitch * (*context).luma_height as usize;
    m.dstDevice = (frame as *mut u8).wrapping_add(offset) as CUdeviceptr;
    m.Height = (*context).chroma_height as usize;
    cuMemcpy2DAsync_v2(&m, &mut *(*context).cuda_stream.cu_stream());

    if (*context).num_chroma_planes == 2 {
        let offset = m.srcPitch * ((((*context).surface_height as usize + 1) & !1) * 2);
        m.srcDevice = (dp_src_frame as *mut u8).wrapping_add(offset) as CUdeviceptr;

        let offset = m.dstPitch * (*context).luma_height as usize * 2;
        m.dstDevice = (frame as *mut u8).wrapping_add(offset) as CUdeviceptr;
        m.Height = (*context).chroma_height as usize;
        cuMemcpy2DAsync_v2(&m, &mut *(*context).cuda_stream.cu_stream());
    }
    unsafe { (DECODE_API.unmap_video_frame)((*context).decoder, dp_src_frame) };

    let cuda_slice = (*context)
        .cuda_stream
        .upgrade_device_ptr(frame, frame_size as usize);

    (*context)
        .frame_queue
        .lock()
        .unwrap()
        .push_back(Arc::new(cuda_slice));

    1
}

unsafe extern "C" fn handle_operating_point(
    arg1: *mut ::core::ffi::c_void,
    p_op_info: *mut CUVIDOPERATINGPOINTINFO,
) -> ::core::ffi::c_int {
    let mut context_box: Box<DecodeContext> = Box::from_raw(arg1 as *mut DecodeContext);
    if (*p_op_info).codec == cudaVideoCodec_enum::cudaVideoCodec_AV1 {
        if (*p_op_info).__bindgen_anon_1.av1.operating_points_cnt > 1 {
            if context_box.operating_pts
                >= ((*p_op_info).__bindgen_anon_1.av1.operating_points_cnt as i32)
            {
                context_box.operating_pts = 0;
            }
            return context_box.operating_pts | ((context_box.display_all_layers as i32) << 10);
        }
    }

    -1
}

impl Decoder {
    ///
    pub fn initialize_with_cuda(
        cuda_ctx: Arc<CudaContext>,
        cuda_stream: Arc<CudaStream>,
        codec_type: cudaVideoCodec_enum,
        resize_info: (i32, i32),
    ) -> Result<Self, DecodeError> {
        let context = Box::new(DecodeContext::new(
            cuda_ctx.clone(),
            cuda_stream.clone(),
            codec_type,
            Dim {
                w: resize_info.0,
                h: resize_info.1,
            },
        ));
        let context: *mut c_void = Box::into_raw(context) as *mut c_void;
        // cuda_device.cu_primary_ctx()

        let mut parser_parameters = CUVIDPARSERPARAMS {
            CodecType: codec_type,
            ulMaxNumDecodeSurfaces: 1,
            ulClockRate: 1000,
            ulMaxDisplayDelay: 0,
            pUserData: context,
            pfnSequenceCallback: Some(handle_video_seq),
            pfnDecodePicture: Some(handle_picture_decode),
            pfnDisplayPicture: Some(handle_picture_display),
            pfnGetOperatingPoint: Some(handle_operating_point),
            ..Default::default()
        };
        let mut parser = ptr::null_mut();
        unsafe { (DECODE_API.create_video_parser)(&mut parser, &mut parser_parameters) };

        Ok(Decoder {
            parser: ParserPtr(parser),
            decoder_context: DecCtxPtr(context),
            _ctx: cuda_ctx,
            _stream: cuda_stream,
        })
    }

    fn get_decoder_context(&self) -> *mut DecodeContext {
        (self.decoder_context.0) as *mut DecodeContext
    }

    ///
    pub fn decode(&mut self, input: *mut c_uchar, size: u64) -> usize {
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags: CUvideopacketflags::CUVID_PKT_TIMESTAMP as u64,
            payload_size: size,
            payload: input,
            timestamp: 0,
        };
        let res = unsafe { (DECODE_API.parse_video_data)(self.parser.0, &mut packet) };
        res.result().unwrap();

        // The callback `handle_picture_decode` should populate this queue with decoded frames
        unsafe {
            (*self.get_decoder_context())
                .frame_queue
                .lock()
                .unwrap()
                .len()
        }
    }

    ///
    pub fn get_frame(&mut self) -> Option<Arc<CudaSlice<f32>>> {
        unsafe {
            (*self.get_decoder_context())
                .frame_queue
                .lock()
                .unwrap()
                .pop_front()
        }
    }
}
