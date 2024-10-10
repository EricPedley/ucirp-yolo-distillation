Gitignored folders:
```
frames
google_images
google_images_2
videos
runs/detect
yolo_output
```

Procedure
1. Put videos in videos folder (tested with webm and mkv formats so far)
2. run `to_frames.py`, which will fill `frames`
3. Run `download_google_images.py` to fill an arbitrary folder with images from google.
4. edit line in `dino.py` that looks like `for folder in ['example_folder_1', ...etc]` to include only new folders
5. run dino.py, which will add images and labels to `yolo_output`

The reason it fills the folder sequentially instead of regenerating everything at once is so I can incrementally add more data and not waste time auto-annotating existing data. I could automate all these steps and make it foolproof but IMO it's not worth the time rn.
