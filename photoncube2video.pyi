from typing import Optional

class PhotonCube:
    @staticmethod
    def convert_to_npy(
        src: str,
        dst: str,
        is_full_array: bool,
        message: Optional[str],
    ) -> None: ...
