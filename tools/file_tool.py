import os
import requests
from configs import PROJECT_DIR
from configs import logger
import hashlib


def save_file_from_url(url, dest_path):
    """
    从给定 URL 下载文件到本地路径。

    示例：
    source_url = "https://example.invalid/path/to/file.bin"
    target_path = "./downloads/file.bin"
    save_file_from_url(source_url, target_path)

    :param url:
    :param dest_path:
    :return:
    """
    logger.info(f"begin to download {url} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_download_path = dest_path + "_download"
    if os.path.exists(tmp_download_path):
        os.remove(tmp_download_path)
    content_length = 0
    if os.path.exists(dest_path):
        os.remove(dest_path)
    try:
        res = requests.get(url, stream=True)
        if res.status_code != 200:
            logger.info(f"请求url出错！ 错误码：{res.status_code}, url：{url}")
            return
        if "content-length" in res.headers:
            content_length = int(res.headers['content-length'])
            # 若当前报文长度小于前次报文长度，或者已接收文件等于当前报文长度，则可以认为视频接收完成
            if (os.path.exists(dest_path) and os.path.getsize(dest_path) == content_length) \
                    or content_length == 0:
                logger.info(f"already exist file,url is {url}, dest path is {dest_path}")
                return
            elif (os.path.exists(dest_path) and os.path.getsize(dest_path) != content_length):
                logger.info('文件尺寸不匹配, 重新下载，file size : %.2f M   total size:%.2f M' % (
                    os.path.getsize(dest_path) / 1024 / 1024, content_length / 1024 / 1024))
                os.remove(dest_path)
        else:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        # 写入收到的视频数据
        with open(tmp_download_path, 'wb') as file:
            for chunk in res.iter_content(chunk_size=10240):
                file.write(chunk)
            file.flush()
        os.rename(tmp_download_path, dest_path)
        logger.info('下载成功,file size : %.2f M   total size:%.2f M' % (
            os.path.getsize(dest_path) / 1024 / 1024, content_length / 1024 / 1024))
    except Exception as e:
        logger.exception(f"fail download video from url {url}")
        if os.path.exists(tmp_download_path):
            os.remove(tmp_download_path)
        raise e


def check_or_download_model_file(urls_and_relative_paths, root_dir=PROJECT_DIR):
    """
    根据下载地址和相对路径检查模型文件是否存在，不存在则下载。
    :param urls_and_relative_paths:
    :param root_dir:
    :return:
    """
    logger.info(f"begin to check model {str(urls_and_relative_paths)}")
    for file_url, relative_path in urls_and_relative_paths:
        abs_path = os.path.join(root_dir, relative_path)
        if os.path.exists(abs_path):
            continue
        else:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            retry = 3
            while retry > 0:
                try:
                    save_file_from_url(file_url, abs_path)
                    break
                except:
                    logger.exception("")
                    logger.info(f"download file fail ,url is {file_url}, retry download, left retry time is {retry}")
                finally:
                    retry -= 1


def compute_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def generate_md5(directory, md5_file):
    """
    为文件夹下的数据生成md5值
    :param directory:
    :param md5_file:
    :return:
    """
    content = ""
    directory = os.path.abspath(directory)
    with open(md5_file, "w") as f:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path) and os.path.abspath(md5_file) != os.path.abspath(file_path):
                    md5_value = compute_md5(file_path)
                    line = f"{md5_value} {os.path.relpath(file_path,directory)}\n"
                    content += line
                    f.write(line)
    logger.info(f"MD5 values saved to {md5_file}, content=\n{content}")


def generate_md5_default(directory):
    md5_file = os.path.join(directory, "checksum.md5")
    generate_md5(directory, md5_file)


def verify_md5(directory, md5_file):
    """
    校验md5值
    :param directory:
    :param md5_file:
    :return:
    """
    not_valid_file_path = []
    with open(md5_file, "r") as f:
        md5_values = f.readlines()
    logger.info("begin md5 check")
    logger.info("################")
    for line in md5_values:
        parts = line.strip().split(" ")
        if len(parts) == 2:
            md5_value, filename = parts
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                current_md5 = compute_md5(file_path)
                if current_md5 == md5_value:
                    logger.info(f"{filename}: OK")
                else:
                    not_valid_file_path.append(os.path.join(directory, filename))
                    logger.info(f"{filename}: FAILED")
            else:
                logger.info(f"{filename}: NOT FOUND")
        else:
            logger.info(f"Invalid line in MD5 file: {line.strip()}")
    logger.info("################")
    return not_valid_file_path


def verify_md5_default(directory):
    md5_file = os.path.join(directory, "checksum.md5")
    if os.path.exists(md5_file):
        return verify_md5(directory, md5_file)
    else:
        return []


if __name__ == '__main__':
    # example_target = os.path.join(PROJECT_DIR, "checkpoints", "PUT_CHECKPOINT_HERE")
    # result = compute_md5(example_target)
    # print(result)
    # generate_md5_default(os.path.join(PROJECT_DIR, "checkpoints"))
    not_valid_file_path = verify_md5_default(os.path.join(PROJECT_DIR, "checkpoints"))
    print(not_valid_file_path)
