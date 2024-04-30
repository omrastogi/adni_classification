import io

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm


# def download_file(real_file_id):
#   """Downloads a file
#   Args:
#       real_file_id: ID of the file to download
#   Returns : IO object with location.

#   Load pre-authorized user credentials from the environment.
#   TODO(developer) - See https://developers.google.com/identity
#   for guides on implementing OAuth2 for the application.
#   """
#   creds, _ = google.auth.default()

#   try:
#     # create drive api client
#     service = build("drive", "v3", credentials=creds)

#     file_id = real_file_id

#     # pylint: disable=maybe-no-member
#     request = service.files().get_media(fileId=file_id)
#     file = io.BytesIO()
#     downloader = MediaIoBaseDownload(file, request)
#     done = False

#     while done is False:
#       status, done = downloader.next_chunk()
#     #   print(f"Download {int(status.progress() * 100)}.")
#       tqdm.write(f"Download {int(status.progress() * 100)}%")

#   except HttpError as error:
#     print(f"An error occurred: {error}")
#     file = None

#   return file.getvalue()


# if __name__ == "__main__":
# #   download_file(real_file_id="1D350iwtcUDytqIe_YL6pcIXjb3774_zm")
#   download_file(real_file_id="1_3L9ZMoM-ovUg3rlBUduop1HdbncsFSj")

#   https://drive.google.com/file/d/1_3L9ZMoM-ovUg3rlBUduop1HdbncsFSj/view?usp=sharing

def download_file_and_save(real_file_id, destination_path):
    creds, _ = google.auth.default()
    try:
        service = build("drive", "v3", credentials=creds)
        request = service.files().get_media(fileId=real_file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        with tqdm(total=100, unit='%', desc='Downloading') as pbar:
            while not done:
                status, done = downloader.next_chunk()
                pbar.update(int(status.progress() * 100) - pbar.n)
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

    # Write the downloaded bytes to a file
    with open(destination_path, 'wb') as f:
        f.write(file.getvalue())
    return destination_path

if __name__ == "__main__":
    # export GOOGLE_APPLICATION_CREDENTIALS="/data2/om/ADNI dataset/credentials.json"
    # Example usage: Specify the path where you want to save the file
    # print("dowloading AD")
    # download_file_and_save(real_file_id="1D350iwtcUDytqIe_YL6pcIXjb3774_zm", destination_path="AD_processed.zip")
    print("downloading CN")
    download_file_and_save(real_file_id="14Uw0lFkbJaeM7aO1CO7eGH_rA5oZPq7S", destination_path="CN_processed.zip")

    # download_file_and_save(real_file_id="1_3L9ZMoM-ovUg3rlBUduop1HdbncsFSj", destination_path="downloaded_file.jpg")
