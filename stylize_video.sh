set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}

# Find out whether ffmpeg or avconv is installed on the system
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
  FFMPEG=avconv
  command -v $FFMPEG >/dev/null 2>&1 || {
    echo >&2 "This script requires either ffmpeg or avconv installed.  Aborting."; exit 1;
  }
}

if [ "$#" -le 1 ]; then
   echo "Usage: bash stylize_video.sh <path_to_video> <path_to_style_image>"
   exit 1
fi

echo ""
read -p "Did you install the required dependencies? [y/n] $cr > " dependencies

if [ "$dependencies" != "y" ]; then
  echo "Error: Requires dependencies: tensorflow, opencv2 (python), scipy"
  exit 1;
fi

echo ""
read -p "Do you have a CUDA enabled GPU? [y/n] $cr > " cuda

#if [ "$cuda" != "y" ]; then
#  echo "Error: GPU required to render videos in a feasible amount of time."
#  exit 1;
#fi

# Parse arguments
content_video="$1"
content_dir=$(dirname "$content_video")
content_filename=$(basename "$content_video")
extension="${content_filename##*.}"
content_filename="${content_filename%.*}"
content_filename=${content_filename//[%]/x}

style_image="$2"
style_dir=$(dirname "$style_image")
style_filename=$(basename "$style_image")

if [ ! -d "./video_input" ]; then
  mkdir -p ./video_input
fi
temp_dir="./video_input/${content_filename}"

# Create output folder
mkdir -p "$temp_dir"

# Save frames of the video as individual image files
$FFMPEG -v quiet -i "$1" "${temp_dir}/frame_%04d.jpg"
eval $(ffprobe -v error -of flat=s=_ -select_streams v:0 -show_entries stream=width,height "$1")
width="${streams_stream_0_width}"
height="${streams_stream_0_height}"
if [ "$width" -gt "$height" ]; then
  max_size="$width"
else
  max_size="$height"
fi
num_frames=$(find "$temp_dir" -iname "*.jpg" | wc -l)

#python3 Deep_Laplacian.py --video_frames_input_dir ./video_input/tenor --laplacian_dir './video_input/laplacian' --video

echo "Computing optical flow [CPU]. This will take a while..."
cd ./video_input
bash make-opt-flow.sh ${content_filename}/frame_%04d.jpg ${content_filename}
cd ..

# Changing dir to tmp
temp_dir="./video_input/tmp/${content_filename}"

#echo "Rendering stylized video frames [CPU & GPU]. This will take a while..."
python neural_style.py --video \
--video_input_dir "${temp_dir}" \
--style_imgs_dir "${style_dir}" \
--style_imgs "${style_filename}" \
--end_frame "${num_frames}" \
--content_frame_frmt "tmp/frame_{}.jpg" \
--max_size "${max_size}" \
--laplacian_dir './video_input/laplacian' \
--verbose;

# Create video from output images.
echo "Converting image sequence to video.  This should be quick..."
$FFMPEG -v quiet -i ./video_output/frame_%04d.jpg ./video_output/${content_filename}-stylized.$extension

# Clean up garbage
if [ -d "${temp_dir}" ]; then
  rm -rf "${temp_dir}"
fi
