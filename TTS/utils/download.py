# Adapted from https://github.com/pytorch/audio/

import hashlib
import logging
import os
import tarfile
import urllib
import urllib.parse
import urllib.request
import zipfile
from os.path import expanduser
from typing import Any, Iterable, List, Optional

from torch.utils.model_zoo import tqdm

URL_TIMEOUT = 60
ALLOWED_URL_SCHEMES = ("http", "https")


def _open_url(req: urllib.request.Request, timeout: int = URL_TIMEOUT):
    """Open an http(s) request with a timeout, refusing file:// and other schemes."""
    scheme = urllib.parse.urlparse(req.full_url).scheme
    if scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(f" [!] Refusing to open URL with scheme '{scheme}': {req.full_url}")
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310  # noqa: S310  scheme checked above


def _resolve_inside(output_folder: str, real_root: str, member_name: str, archive_name: str) -> str:
    """Return the real path a member would land on, or raise if it escapes the output folder."""
    target = os.path.realpath(os.path.join(output_folder, member_name))
    if os.path.commonpath([real_root, target]) != real_root:
        raise ValueError(f" [!] Archive member '{member_name}' in {archive_name} resolves outside the output folder.")
    return target


def _check_tar_member(member: tarfile.TarInfo, output_folder: str, real_root: str, archive_name: str) -> None:
    _resolve_inside(output_folder, real_root, member.name, archive_name)
    if member.issym() or member.islnk():
        # Hard links are relative to the archive root while symlinks are relative to the member's directory.
        link_base = "" if member.islnk() else os.path.dirname(member.name)
        link_path = member.linkname if os.path.isabs(member.linkname) else os.path.join(link_base, member.linkname)
        _resolve_inside(output_folder, real_root, link_path, f"{archive_name} (link target of '{member.name}')")
    if member.isdev():
        raise ValueError(f" [!] Archive member '{member.name}' in {archive_name} is a device node.")


def _check_zip_member(member_name: str, output_folder: str, real_root: str, archive_name: str) -> None:
    if os.path.isabs(member_name) or member_name.startswith(("/", "\\")):
        raise ValueError(f" [!] Archive member '{member_name}' in {archive_name} has an absolute path.")
    _resolve_inside(output_folder, real_root, member_name, archive_name)


def safe_extract_archive(archive_path: str, output_folder: str) -> List[str]:
    """Extract a zip or tar archive into `output_folder`, rejecting members that would land outside it.

    Args:
        archive_path (str): path to the zip or tar archive.
        output_folder (str): folder to extract into.

    Returns:
        list: member names in the archive, in archive order.
    """
    real_root = os.path.realpath(output_folder)
    archive_name = os.path.basename(archive_path)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zfile:
            names = zfile.namelist()
            for name in names:
                _check_zip_member(name, output_folder, real_root, archive_name)
            for name in names:
                zfile.extract(name, output_folder)
        return names
    if tarfile.is_tarfile(archive_path):
        # Every member is validated before extraction and the data filter is applied.
        with tarfile.open(archive_path) as tar:  # nosemgrep
            members = tar.getmembers()
            for member in members:
                _check_tar_member(member, output_folder, real_root, archive_name)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(output_folder, filter="data")
            else:
                # Older interpreters have no extraction filter, so each member is extracted only after its own check.
                for member in members:
                    tar.extract(member, output_folder)
        return [member.name for member in members]
    raise ValueError(f" [!] {archive_path} is not a zip or tar archive.")


def stream_url(
    url: str, start_byte: Optional[int] = None, block_size: int = 32 * 1024, progress_bar: bool = True
) -> Iterable:
    """Stream url by chunk

    Args:
        url (str): Url.
        start_byte (int or None, optional): Start streaming at that point (Default: ``None``).
        block_size (int, optional): Size of chunks to stream (Default: ``32 * 1024``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
    """

    # If we already have the whole file, there is no need to download it again
    req = urllib.request.Request(url, method="HEAD")  # noqa: S310  opened through _open_url, which checks the scheme
    with _open_url(req) as response:
        url_size = int(response.info().get("Content-Length", -1))
    if url_size == start_byte:
        return

    req = urllib.request.Request(url)  # noqa: S310  opened through _open_url, which checks the scheme
    if start_byte:
        req.headers["Range"] = "bytes={}-".format(start_byte)

    with (
        _open_url(req) as upointer,
        tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=url_size,
            disable=not progress_bar,
        ) as pbar,
    ):
        num_bytes = 0
        while True:
            chunk = upointer.read(block_size)
            if not chunk:
                break
            yield chunk
            num_bytes += len(chunk)
            pbar.update(len(chunk))


def download_url(
    url: str,
    download_folder: str,
    filename: Optional[str] = None,
    hash_value: Optional[str] = None,
    hash_type: str = "sha256",
    progress_bar: bool = True,
    resume: bool = False,
) -> None:
    """Download file to disk.

    Args:
        url (str): Url.
        download_folder (str): Folder to download file.
        filename (str or None, optional): Name of downloaded file. If None, it is inferred from the url
            (Default: ``None``).
        hash_value (str or None, optional): Hash for url (Default: ``None``).
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
        resume (bool, optional): Enable resuming download (Default: ``False``).
    """

    req = urllib.request.Request(url, method="HEAD")  # noqa: S310  opened through _open_url, which checks the scheme
    with _open_url(req) as response:
        req_info = response.info()

    # Detect filename
    filename = filename or req_info.get_filename() or os.path.basename(url)
    filepath = os.path.join(download_folder, filename)
    if resume and os.path.exists(filepath):
        mode = "ab"
        local_size: Optional[int] = os.path.getsize(filepath)

    elif not resume and os.path.exists(filepath):
        raise RuntimeError("{} already exists. Delete the file manually and retry.".format(filepath))
    else:
        mode = "wb"
        local_size = None

    if hash_value and local_size == int(req_info.get("Content-Length", -1)):
        with open(filepath, "rb") as file_obj:
            if validate_file(file_obj, hash_value, hash_type):
                return
        raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(filepath))

    with open(filepath, mode) as fpointer:
        for chunk in stream_url(url, start_byte=local_size, progress_bar=progress_bar):
            fpointer.write(chunk)

    with open(filepath, "rb") as file_obj:
        if hash_value and not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(filepath))


def validate_file(file_obj: Any, hash_value: str, hash_type: str = "sha256") -> bool:
    """Validate a given file object with its hash.

    Args:
        file_obj: File object to read from.
        hash_value (str): Hash for url.
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).

    Returns:
        bool: return True if its a valid file, else False.
    """

    if hash_type == "sha256":
        hash_func = hashlib.sha256()
    elif hash_type == "md5":
        hash_func = hashlib.md5(usedforsecurity=False)
    else:
        raise ValueError

    while True:
        # Read by chunk to avoid filling memory
        chunk = file_obj.read(1024**2)
        if not chunk:
            break
        hash_func.update(chunk)

    return hash_func.hexdigest() == hash_value


def extract_archive(from_path: str, to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    """Extract archive.
    Args:
        from_path (str): the path of the archive.
        to_path (str or None, optional): the root path of the extraced files (directory of from_path)
            (Default: ``None``)
        overwrite (bool, optional): overwrite existing files (Default: ``False``)

    Returns:
        list: List of paths to extracted files even if not overwritten.
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)
    real_root = os.path.realpath(to_path)
    archive_name = os.path.basename(from_path)

    try:
        with tarfile.open(from_path, "r") as tar:
            logging.info("Opened tar file %s.", from_path)
            # Every member is checked before anything is written so a bad member late in the archive aborts cleanly.
            members = tar.getmembers()
            for member in members:
                _check_tar_member(member, to_path, real_root, archive_name)
            files = []
            for file_ in members:  # type: Any
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("%s already extracted.", file_path)
                        if not overwrite:
                            continue
                if hasattr(tarfile, "data_filter"):
                    tar.extract(file_, to_path, filter="data")
                else:
                    tar.extract(file_, to_path)
            return files
    except tarfile.ReadError as e:
        logging.debug("%s is not a tar archive (%s), trying zip.", from_path, e)

    try:
        with zipfile.ZipFile(from_path, "r") as zfile:
            logging.info("Opened zip file %s.", from_path)
            files = zfile.namelist()
            for file_ in files:
                _check_zip_member(file_, to_path, real_root, archive_name)
            for file_ in files:
                file_path = os.path.join(to_path, file_)
                if os.path.exists(file_path):
                    logging.info("%s already extracted.", file_path)
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        return files
    except zipfile.BadZipFile as e:
        logging.debug("%s is not a zip archive (%s).", from_path, e)

    raise NotImplementedError(" > [!] only supports tar.gz, tgz, and zip achives.")


def download_kaggle_dataset(dataset_path: str, dataset_name: str, output_path: str):
    """Download dataset from kaggle.
    Args:
        dataset_path (str):
        This the kaggle link to the dataset. for example vctk is 'mfekadu/english-multispeaker-corpus-for-voice-cloning'
        dataset_name (str): Name of the folder the dataset will be saved in.
        output_path (str): Path of the location you want the dataset folder to be saved to.
    """
    data_path = os.path.join(output_path, dataset_name)
    try:
        import kaggle  # pylint: disable=import-outside-toplevel

        kaggle.api.authenticate()
        print(f"""\nDownloading {dataset_name}...""")
        kaggle.api.dataset_download_files(dataset_path, path=data_path, unzip=True)
    except OSError:
        print(
            f"""[!] in order to download kaggle datasets, you need to have a kaggle api token stored in your {os.path.join(expanduser('~'), '.kaggle/kaggle.json')}"""
        )
