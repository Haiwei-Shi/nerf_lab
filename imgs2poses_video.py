from llff.poses.pose_utils import gen_poses
import sys
import os
import subprocess
import argparse

def extract_frames(video_path, output_dir, fps=5):
	"""Extract frames from a video using FFmpeg."""
	os.makedirs(output_dir, exist_ok=True)
	cmd = f"ffmpeg -i \"{video_path}\" -qscale:v 1 -qmin 1 -vf \"fps={fps}\" \"{output_dir}/%04d.jpg\""
	try:
		subprocess.run(cmd, shell=True, check=True)
		print(f"Extracted frames saved in {output_dir}")
	except subprocess.CalledProcessError:
		print("Error: FFmpeg failed to extract frames.")
		sys.exit(1)




if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--match_type', type=str, 
						default='exhaustive_matcher', help='type of matcher used.  Valid options: \
						exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
	parser.add_argument("--input", type=str, required=True, help="Path to video file or image directory.")
	parser.add_argument("--output", type=str, required=True, help="Path to save processed LLFF data.")
	parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract from video.")
	args = parser.parse_args()


	if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
		print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
		sys.exit()

	image_dir = os.path.join(args.output, "images")
	os.makedirs(args.output, exist_ok=True)

	# Extract frames from video
	extract_frames(args.input, image_dir, fps=args.fps)

	print(f"Processing images in {image_dir} and saving results to {args.output}")

	# Run gen_poses() to process images and compute camera poses
	gen_poses(args.output, args.match_type)
