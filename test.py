import ffmpeg
import numpy as np
import cv2

def get_frame(input_file, stream_index=1):
   process = (
       ffmpeg
       .input(input_file)
       .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
       .global_args('-map', f'0:{stream_index}')
       .run_async(pipe_stdout=True)
   )
   
   in_bytes = process.stdout.read(1920*1080*3)  # Adjust size based on resolution
   frame = np.frombuffer(in_bytes, np.uint8).reshape([-1, 1920, 3])
   
   process.stdout.close()
   process.wait()
   
   return frame
   
while True:
    input_file = "/Volumes/Cameras/aqara_video/lumi1.54ef44457bc9/20250207/030000.mp4"
    
    frame = get_frame(input_file)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

