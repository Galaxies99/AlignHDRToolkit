from models import aligner, merger, tonemapper


# HDR synthesis
def hdrSynthesis(images, exposures, ref_id=None,
                 aligner=aligner.AlignFeatureORB(nfeatures=50000, min_matches=1000, match_percent=0.05),
                 merger=merger.MergerDebevec()):
    if images is [] or exposures is []:
        return None
    images = aligner.process(images, ref_id)
    hdr = merger.process(images, exposures)
    return hdr


# Tonemapping HDR image to LDR image
def hdrTonemapping(hdr, tonemapper=tonemapper.TonemapperReinhard(gamma=1.5, intensity=0, light_adapt=0, color_adapt=0)):
    return tonemapper.process(hdr)
