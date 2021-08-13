from adet.config import get_cfg

def prepare_cfg(cfg_file, model_path):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(['MODEL.WEIGHTS', model_path])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg