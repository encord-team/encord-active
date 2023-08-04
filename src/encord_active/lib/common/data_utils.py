import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import as_completed
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Generator, Optional, Sequence, TypeVar
from urllib.parse import unquote, urlparse

import numpy as np
import requests
from encord.exceptions import EncordException, UnknownException
from PIL import Image
from tqdm.auto import tqdm

_EXTRACT_FRAMES_CACHE: Dict[str, int] = {}
_EXTRACT_FRAMES_FOLDER: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()


def extract_frames(video_file_name: Path, img_dir: Path, data_hash: str, symlink_folder: bool = True) -> None:
    if data_hash not in _EXTRACT_FRAMES_CACHE:
        while True:
            try:
                cache_id = len(_EXTRACT_FRAMES_CACHE)
                tempdir = Path(_EXTRACT_FRAMES_FOLDER.name) / f"extra_cache_{cache_id}"
                tempdir.mkdir()
                break
            except FileExistsError:
                time.sleep(0.001)
                pass

        try:
            _extract_frames(video_file_name, tempdir, data_hash)
        except Exception:
            shutil.rmtree(tempdir, ignore_errors=True)
            raise
        _EXTRACT_FRAMES_CACHE[data_hash] = cache_id
    # Symlink everything in the temporary directory, do not do duplicate work
    read_cache_id = _EXTRACT_FRAMES_CACHE[data_hash]
    read_tempdir = Path(_EXTRACT_FRAMES_FOLDER.name) / f"extra_cache_{read_cache_id}"
    if symlink_folder:
        img_dir.symlink_to(read_tempdir, target_is_directory=True)
    else:
        img_dir.mkdir(parents=True, exist_ok=True)
        for frame in read_tempdir.iterdir():
            (img_dir / frame.name).symlink_to(frame, target_is_directory=False)


def _extract_frames(video_file_name: Path, img_dir: Path, data_hash: str) -> None:
    # DENIS: for the rest to work, I will need to throw if the current directory exists and give a nice user warning.
    img_dir.mkdir(parents=True, exist_ok=True)
    command = f'ffmpeg -i "{video_file_name}" -start_number 0 {img_dir}/{data_hash}_%d.png -hide_banner'
    if subprocess.run(command, shell=True, capture_output=True, stdout=None, check=False).returncode != 0:
        raise RuntimeError(
            "Failed to split the video into multiple image files. Please ensure that you have FFMPEG "
            f"installed on your machine. You can download it from https://ffmpeg.org/download.html. "
            f" The command that failed was: `{command}`."
        )


def count_frames(video_file_name: Path) -> int:
    command = f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "{video_file_name}"'
    output = subprocess.run(command, shell=True, capture_output=True, stdout=None, check=False)
    if output.returncode != 0:
        raise RuntimeError(
            "Failed to count the number of frames in the video. Please ensure that you have FFMPEG "
            "installed on your machine. You can download it from https://ffmpeg.org/download.html. "
            f"The command that failed was: `{command}`."
        )
    output_str = output.stdout.decode("utf-8")
    return int(output_str)


def get_frames_per_second(video_file_name: Path) -> float:
    command = f'ffmpeg -i "{video_file_name}" 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"'
    output = subprocess.run(command, shell=True, capture_output=True, stdout=None, check=False)
    if output.returncode != 0:
        raise RuntimeError(
            "Failed to count the frame rate in the video. Please ensure that you have FFMPEG "
            f"installed on your machine. You can download it from https://ffmpeg.org/download.html. "
            f"The command that failed was: `{command}`."
        )
    output_str = output.stdout.decode("utf-8")
    return float(output_str)


_DOWNLOAD_CACHE: Dict[str, int] = {}
_DOWNLOAD_CACHE_FOLDER: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()


def _add_to_cache(key: str) -> Path:
    if key in _DOWNLOAD_CACHE:
        raise RuntimeError(f"Duplicate cache write for : {key}")
    new_id = len(_DOWNLOAD_CACHE)
    _DOWNLOAD_CACHE[key] = new_id
    path = Path(_DOWNLOAD_CACHE_FOLDER.name) / f"cache_{new_id}"
    if path.exists():
        raise RuntimeError(f"Local temp-file cache bug: {key}")
    return path


def _remove_from_cache(key: str) -> None:
    cache_id = _DOWNLOAD_CACHE[key]
    cache_path = Path(_DOWNLOAD_CACHE_FOLDER.name) / f"cache_{cache_id}"
    if cache_path.exists():
        # Cleanup, so it can be used again
        os.remove(cache_path)
    del _DOWNLOAD_CACHE[key]


def _get_from_cache(key: str) -> Optional[Path]:
    cache_id = _DOWNLOAD_CACHE.get(key, None)
    if cache_id is not None:
        return Path(_DOWNLOAD_CACHE_FOLDER.name) / f"cache_{cache_id}"
    return None


def download_file(
    url: str,
    project_dir: Path,
    destination: Path,
    byte_size: int = 1024,
    cache: bool = True,
) -> Path:
    if destination.is_file():
        return destination

    url_path = url_to_file_path(url, project_dir)
    if url_path is not None:
        if cache:
            destination.symlink_to(url_path)
        else:
            shutil.copyfile(url_path, destination)
        return destination

    cached_download = _get_from_cache(url)
    if cached_download is not None:
        if cache:
            destination.symlink_to(cached_download)
        else:
            shutil.copyfile(cached_download, destination)
        return destination

    if cache:
        download_target = _add_to_cache(url)
    else:
        download_target = destination
    try:
        with open(download_target, "xb") as file:
            r = requests.get(url, stream=True)

            if r.status_code != 200:
                raise ConnectionError(f"Something happened, couldn't download file from: {url}")

            for chunk in r.iter_content(chunk_size=byte_size):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
            file.flush()
    except Exception:
        if cache:
            _remove_from_cache(url)
        raise
    if cache:
        destination.symlink_to(download_target)
    return destination


def url_to_file_path(url: str, project_dir: Path) -> Optional[Path]:
    if url.startswith(("/", "./", "../", "~/")):
        return Path(url)
    if url.startswith("file:"):
        return Path(unquote(urlparse(url).path))
    if url.startswith("relative://"):
        relative_path = url[len("relative://") :]
        return project_dir / Path(relative_path)
    if url.startswith("absolute://"):
        absolute_path = url[len("absolute:/") :]
        return Path(absolute_path)
    return None


def file_path_to_url(path: Path, project_dir: Path, relative: Optional[bool] = None) -> str:
    path = path.expanduser().absolute().resolve()
    if relative is not None and not relative:
        return path.as_uri()

    # Attempt to create relative path
    root_path = project_dir.expanduser().absolute().resolve()
    try:
        rel_path = path.relative_to(root_path)
        return f"relative://{rel_path.as_posix()}"
    except ValueError:
        if relative is not None:
            raise

    # Fallback to strict uri
    return path.as_uri()


def download_image(url: str, project_dir: Path) -> Image.Image:
    path = url_to_file_path(url, project_dir)
    if path is not None:
        return Image.open(path)

    cached_image = _get_from_cache(url)
    if cached_image is not None:
        return Image.open(cached_image)

    new_cache_download = _add_to_cache(url)
    try:
        r = requests.get(url)

        if r.status_code != 200:
            raise ConnectionError(f"Something happened, couldn't download file from: {url}")

        new_cache_download.write_bytes(r.content)
    except Exception:
        _remove_from_cache(url)
        raise
    return Image.open(BytesIO(r.content))


def convert_image_bgr(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    np_image = np.array(rgb_image)
    ocv_image = np_image[:, :, ::-1].copy()
    return ocv_image


TType = TypeVar("TType")


def iterate_in_batches(seq: Sequence[TType], size: int) -> Generator[Sequence[TType], None, None]:
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def collect_async(fn, job_args, max_workers=min(10, (os.cpu_count() or 1) + 4), **kwargs):
    """
    Distribute work across multiple workers. Good for, e.g., downloading data.
    Will return results in dictionary.
    :param fn: The function to be applied
    :param job_args: Arguments to `fn`.
    :param max_workers: Number of workers to distribute work over.
    :param kwargs: Arguments passed on to tqdm.
    :return: List [fn(*job_args)]
    """
    job_args = list(job_args)
    if len(job_args) == 0:
        return []
    if not isinstance(job_args[0], tuple):
        job_args = [(j,) for j in job_args]

    results = []
    with tqdm(total=len(job_args), **kwargs) as pbar:
        with Executor(max_workers=max_workers) as exe:
            jobs = [exe.submit(fn, *args) for args in job_args]
            for job in as_completed(jobs):
                result = job.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
    return results


def try_execute(func: Callable, num_tries: int, kwargs=None):
    """
    Try to execute func num_tries, catching connection related exceptions.
    :param func: The function to execute.
    :param num_tries: The number of times to try and execute the connection.
    :param kwargs: A kwargs dict to pass as function arguments.
    :return: The result of func, so func(kwargs).
    """
    for n in range(num_tries):
        try:
            if kwargs:
                return func(**kwargs)
            else:
                return func()
        except (ConnectionError, ConnectionResetError, OSError, UnknownException, EncordException) as e:
            logging.warning(
                f"Handling {e} when executing {func} with args {kwargs}.\n" f" Trying again, attempt number {n + 1}."
            )
            time.sleep(0.5 * num_tries)  # linear backoff
    raise Exception("Reached maximum number of execution attempts.")
