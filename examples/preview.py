import argparse
from pathlib import Path 
from photoncube2video import PhotonCube, Transform

def main(args):
    pc = PhotonCube.open(args.input)
    pc.set_transforms([Transform.Rot90, Transform.FlipUD])
    pc.set_range(0, 10000, 256)
    pc.save_video(
        args.output,
        message="Making video..." 
    ) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Preview PhotonCube")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("output.mp4"))

    args = parser.parse_args()
    main(args)
