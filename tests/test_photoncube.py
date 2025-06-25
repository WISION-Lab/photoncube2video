from natsort import natsorted
import pytest
import numpy as np
import imageio.v3 as iio


@pytest.fixture
def photoncube(tmp_path):
    path = tmp_path / "cube.npy"
    cube = np.random.randint(low=0, high=255, size=(256, 64 // 8, 64), dtype=np.uint8)
    np.save(path, cube)
    return path, cube


def test_import():
    import photoncube2video
    from photoncube2video import PhotonCube, Transform


def test_open(photoncube):
    from photoncube2video import PhotonCube

    path, cube = photoncube
    pc = PhotonCube.open(path)
    assert len(pc) == len(cube)
    assert pc.shape == cube.shape


def test_save_images(photoncube, tmp_path):
    from photoncube2video import PhotonCube

    path, cube = photoncube
    pc = PhotonCube.open(str(path))
    pc.set_range(200, 205, 1)
    pc.save_images(tmp_path)

    for i, path in zip(range(200, 205), natsorted(tmp_path.glob("*.png"))):
        arr = iio.imread(path)
        packed = np.packbits(arr, axis=1).mean(axis=2)
        assert np.allclose(packed, cube[i])


def test_masks_readonly(photoncube):
    from photoncube2video import PhotonCube

    path, _ = photoncube
    pc = PhotonCube.open(path)

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
