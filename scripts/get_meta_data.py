from exiftool import ExifToolHelper
with ExifToolHelper() as et:
    mp4_path = '/media/obin/36724ed6-bcb5-4555-abd1-5a15b9d076bd/40clean/bill_collection_organized/demos/demo1/raw_video.mp4'
    meta = list(et.get_metadata(str(mp4_path)))[0]
    print("")