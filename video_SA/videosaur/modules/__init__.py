from video_SA.videosaur.modules import timm
from video_SA.videosaur.modules.decoders import build as build_decoder
from video_SA.videosaur.modules.encoders import build as build_encoder
from video_SA.videosaur.modules.groupers import build as build_grouper
from video_SA.videosaur.modules.presence_nn import build as build_presence_nn
from video_SA.videosaur.modules.initializers import build as build_initializer
from video_SA.videosaur.modules.networks import build as build_network
from video_SA.videosaur.modules.utils import Resizer, SoftToHardMask
from video_SA.videosaur.modules.utils import build as build_utils
from video_SA.videosaur.modules.utils import build_module, build_torch_function, build_torch_module
from video_SA.videosaur.modules.video import LatentProcessor, MapOverTime, ScanOverTime,IterOverTime,MapOverTime_mask,MapOverTime2,IterOverTime_mask
from video_SA.videosaur.modules.video import build as build_video

__all__ = [
    "build_decoder",
    "build_encoder",
    "build_grouper",
    "build_initializer",
    "build_network",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "MapOverTime_mask",
    "ScanOverTime",
    "LatentProcessor",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "presence_nn": build_presence_nn,
    "initializers": build_initializer,
    "networks": build_network,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
}
