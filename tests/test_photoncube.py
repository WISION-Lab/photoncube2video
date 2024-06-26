import pytest


def test_import():
    import photoncube2video
    from photoncube2video import PhotonCube, Transform


def test_masks_readonly():
    from photoncube2video import PhotonCube

    pc = PhotonCube.open("../aquisitions/binary.npy")

    assert pc.inpaint_mask == None
    assert pc.cfa_mask == None

    pc.load_mask("rgbw_oh_bn_color_ss2_corrected.png")

    assert not pc.inpaint_mask.flags["WRITEABLE"]

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        pc.inpaint_mask[0, 0] = False


def test_validate_transforms():
    from photoncube2video import Transform

    with pytest.raises(RuntimeError, match="invalid variant"):
        Transform.from_str("abc")
