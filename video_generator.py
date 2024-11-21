import gradio as gr
from moviepy.editor import *
from PIL import Image
import tempfile
import utils
import ast
import numpy as np
from moviepy.audio.fx.all import volumex, audio_loop

# Function to create the storyboard
def generate_story(category, product_name):
    try:
        # Generate Storyboard
        scene_list = utils.generate_storyboard(category, product_name)
        scene_list = ast.literal_eval(scene_list)  # Safely evaluate the storyboard
    except Exception as e:
        return f"Error generating storyboard: {e}"
    return scene_list

# Function to generate the video
def generate_video(scene_list, bg_music, logo):
    
    fade_duration = 1
    EFFECT_DURATION = 5
    total_duration = 0
    temp_video_files = []

    for scene_no, scene in enumerate(scene_list):
        # Transition and scene audio
        effect = scene.get("transition", "fade")
        audio_file_path, audio_duration = utils.text_to_speech(scene, scene_no)  # Placeholder for text-to-speech function
        print("Audio Generation Completed !")
        total_duration += audio_duration

        # Generate image or GIF for the scene
        gif_output_path = utils.get_generated_image(scene_no, scene,audio_duration)  # Placeholder for image generation

        # Create video clip for the scene
        img_clip = (VideoFileClip(gif_output_path)
                    .set_duration(audio_duration)
                    .set_position(("center", "center"))
                    .crossfadein(fade_duration))
        img_clip = utils.apply_zoom_in_effect(img_clip, 0.01)  # Apply zoom effect

        # Add audio and logo
        audioclip = AudioFileClip(audio_file_path)
        # Add logo if provided
        if logo is not None:
            logo_clip = (ImageClip(logo.name)
                         .set_duration(audio_duration)
                         .resize(height=70)
                         .margin(right=8, top=8, opacity=0)
                         .set_pos(("right", "top")))
        else:
            logo_clip = None  # No logo clip

        # Compose the scene
        if logo_clip:
            final_clip = CompositeVideoClip([img_clip, logo_clip])
        else:
            final_clip = img_clip
        

        # Compose the scene
        final_clip = CompositeVideoClip([img_clip, logo_clip])
        final_clip.audio = audioclip
        temp_video_path = f'./temp_scene_{scene_no}.mp4'
        final_clip.write_videofile(temp_video_path, fps=30)
        temp_video_files.append(temp_video_path)


    print("Combine all scenes into a single video")
    clips = [VideoFileClip(file).crossfadein(fade_duration).crossfadeout(fade_duration) for file in temp_video_files]
    final_video = concatenate_videoclips(clips, method="compose")

    # Add background music
    # Load the background music and adjust its volume
     # Set the path for the default audio
    default_audio_path = '/home/jupyter/zee_ad_studio/final_vj_audio.wav'

    # Use the uploaded audio if provided, otherwise fall back to the default
    audio_path = bg_music if bg_music is not None else default_audio_path
    
    background_music = AudioFileClip(audio_path).fx(volumex, 0.1)

    # Check the durations
    video_duration = final_video.duration
    music_duration = background_music.duration

    if music_duration < video_duration:
        # Loop the audio if it's shorter than the video's duration
        final_music = audio_loop(background_music, duration=video_duration)
    else:
        # Trim the audio if it's longer than the video's duration
        final_music = background_music.subclip(0, video_duration)

    # Combine the audio tracks
    final_audio = CompositeAudioClip([final_video.audio, final_music])
    final_video.audio = final_audio

    # Export the final video
    output_path = './final_video.mp4'
    final_video.write_videofile(output_path, fps=30)
    return output_path


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Automated Video Generator")
    gr.Markdown("Generate videos based on a storyboard with background music and logos.")

    # Input for story generation
    with gr.Row():
        category = gr.Textbox(label="Category", placeholder="Enter category", value="Default Category")
        product_name = gr.Textbox(label="Product Name", placeholder="Enter product name", value="Default Product")
    
    generate_story_button = gr.Button("Generate Storyboard")

    storyboard_output = gr.JSON(label="Generated Storyboard")
    
    
    generate_story_button.click(
        fn=generate_story,
        inputs=[category, product_name],
        outputs=storyboard_output
    )


    # Input for video generation
    with gr.Row():
        bg_music = gr.File(label="Upload Background Music", file_types=[".mp3"])
        logo = gr.File(label="Upload Logo", file_types=[".png"])
    
    generate_video_button = gr.Button("Generate Video")
    video_output = gr.Video(label="Final Video")
    
    generate_video_button.click(
        fn=generate_video,
        inputs=[storyboard_output, bg_music, logo],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(share=True)