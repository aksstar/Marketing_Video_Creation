import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from google.cloud import texttospeech
from pydub import AudioSegment
from vertexai.preview.vision_models import ImageGenerationModel
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import math
import random
import numpy as np
import imageio
import ast
import glob
from PIL import Image

# Initialize Vertex AI and Text-to-Speech clients
PROJECT_ID = "aakash-test-env"
LOCATION = "us-central1"

def init_vertex_ai():
    """Initialize the Vertex AI environment."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)

# Function to generate storyboard
def generate_storyboard(category, product):
    """
    Generate a detailed storyboard for an advertisement.

    Args:
        text1 (str): Template string for the storyboard prompt.
        category (str): The category of the advertisement.
        product (str): The product featured in the advertisement.

    Returns:
        str: Generated storyboard text.
    """
    init_vertex_ai()
    model = GenerativeModel("gemini-1.5-pro-002")
    text1 = """\"Create a detailed video storyboard for a {category} advertisement featuring the {product}. The video should consist of 6 scenes, each with specific camera movements, text on video, narration, and transition effects. For each scene, please provide the following elements:

Scene Description: A brief description of what happens in the scene. Do not include person/human in the scene as it will be used for image generation

Camera Movement Description: Details on how the camera should move (e.g., panning, zooming, etc.).

Text on Video: The specific text that should appear on the screen, should be a list of 3 to 5 features
text_effect :  it should be one the following listed values [‘fade’, ’scroll’, ‘pop’, ’slide’, ‘typewriter’, ‘bounce’,’wave\']

Narration Text: The script or narration for the scene. this should be atleast 1 to 3 sentences with minimum 50 words

Narrator Tone: The tone of the narrator\'s voice (e.g., \'energetic\', \'informative\', \'warm\', \'exciting\').

Narrator Gender: Specify whether the narrator should be male or female.

Transition: Type of transition between scenes (e.g., \'fade\', \'wipe\', \'zoom\', \'dissolve\').

Generate answers in the form of list of dictonary having keys [\\\'scene_number\\\', \\\'Scene Description\\\', \\\'camera_movement_description\\\', \\\'Text on Video\\\', \\\'Narration Text\\\', \\\'narrator_tone\\\', \\\'transition\\\', \\\'Narrator Gender\\\', \\\'text_effect\\\']\"\"\""""

    formatted_text = text1.format(category=category, product=product)
 

    # Log the generated prompt
    print(f"Generating storyboard with prompt: {formatted_text}")
    
    responses = model.generate_content(
        [formatted_text],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
            "response_mime_type": "application/json",
        },
        safety_settings=[
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.OFF),
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.OFF),
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
        ],
        stream=False,
    )
    
    # Log the response
    print(f"Storyboard generated: {responses.text}")
    return responses.text


# Function to generate an image
def generate_image(prompt, output_file_path):
    """
    Generate an image based on a text prompt.

    Args:
        prompt (str): Description of the desired image.
        output_file_path (str): Path to save the generated image.
    """
    init_vertex_ai()
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
    
    # Log the image generation process
    print(f"Generating image with prompt: {prompt}")
    
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        language="en",
        aspect_ratio="1:1",
        safety_filter_level="block_few",
    )
    
    # Save the generated image
    images[0].save(location=output_file_path, include_generation_parameters=False)
    print(f"Image saved to {output_file_path}")

# Function to get audio duration using PyDub
def get_audio_duration(file_path):
    """
    Calculate the duration of an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        float: Duration in seconds.
    """
    audio_file = AudioSegment.from_file(file_path)
    return audio_file.duration_seconds

# Function to convert text to speech
def text_to_speech(scene, scene_no):
    """
    Convert text to speech for a given scene.

    Args:
        scene (int): Scene number.
        scene_list (list): List of scenes with narration text.

    Returns:
        tuple: Path to the generated audio file and its duration.
    """
    text = scene["Narration Text"]
    gender = scene["Narrator Gender"]
    
    # Log the text-to-speech process
    print(f"Generating audio for scene {scene_no}: {text} with narrator gender {gender}")
    
    # Select voice model based on gender
    voice_model = "en-IN-Standard-B" if gender == "Male" else "en-IN-Standard-A"
    
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name=voice_model)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=1)

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    
    output_file_name = f"output_{scene_no}.mp3"
    output_file_path = f"./audio/{output_file_name}"
    
    with open(output_file_path, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to file {output_file_name}")
    
    duration = get_audio_duration(output_file_path)
    return output_file_path, duration

# Function to apply zoom-in effect on a video clip
def apply_zoom_in_effect(clip, zoom_factor=0.04, duration=1, stop_after=3):
    """
    Apply a zoom-in effect to a video clip.

    Args:
        clip (VideoFileClip): Video clip to apply the effect.
        zoom_factor (float): Zoom factor.
        duration (float): Duration for zoom effect.
        stop_after (float): Stop the zoom effect after this time.

    Returns:
        VideoFileClip: Video clip with zoom-in effect applied.
    """
    # Log the application of zoom effect
    print(f"Applying zoom-in effect with zoom_factor={zoom_factor}, duration={duration}, stop_after={stop_after}")
    
    return clip.resize(lambda t: 1 + zoom_factor * min(t, stop_after) / duration)

# Function to create animations with various text effects over a static image
def create_feature_animation(image_path, features, output_path, effect, audio_duration):
    """
    Create an animated GIF with text overlays and effects.

    Parameters:
    - image_path: Path to the base image.
    - features: List of text features to overlay on the image.
    - output_path: Path where the animated GIF will be saved.
    - effect: Animation effect to apply to the text (e.g., "fade", "scroll", etc.).
    - audio_duration: Duration of the audio (used to sync the animation).
    """
    # Adjust audio duration to leave some buffer
    audio_duration = audio_duration - 3

    # Load the image and resize to standard dimensions
    car_image = Image.open(image_path).resize((800, 500))

    # Load a default font
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    # Settings for text boxes and layout
    box_padding = 10
    box_margin = 10
    text_color = "white"
    box_color = (0, 0, 0, 150)  # Semi-transparent black for text boxes
    start_x = 75
    start_y = 75
    box_width = 300
    feature_height = 40
    spacing = feature_height + box_margin
    frames = []

    # Ensure features are provided
    if not features:
        print(f"No features to animate for {image_path}")
        return []

    print(f"Creating animation with {len(features)} features and effect: {effect}")

    # Function for dynamic color transitions
    def get_fancy_color(alpha):
        r = int(255 * (alpha / 255))
        g = int(255 * ((255 - alpha) / 255))
        b = random.randint(100, 255)  # Add a randomized blue component for variety
        return (r, g, b)

    # Handle different text effects
    if effect == "fade":
        # Gradually fade text into view
        num_fade_steps = int(audio_duration * 10)  # 10 steps per second
        visible_features = []
        for i, feature in enumerate(features):
            visible_features.append(feature)
            for alpha in range(num_fade_steps + 1):
                frame = car_image.copy()
                draw = ImageDraw.Draw(frame)
                for j, visible_feature in enumerate(visible_features):
                    x = start_x
                    y = start_y + j * spacing
                    opacity = int(255 * (alpha / num_fade_steps)) if visible_feature == feature else 255
                    box_opacity = int(150 * (alpha / num_fade_steps)) if visible_feature == feature else 150

                    # Create a semi-transparent overlay for the text box
                    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    box_shape = [(x, y), (x + box_width, y + feature_height)]
                    overlay_draw.rounded_rectangle(box_shape, radius=10, fill=(0, 0, 0, box_opacity))
                    text_x = x + box_padding
                    text_y = y + box_padding
                    overlay_draw.text((text_x, text_y), visible_feature, fill=(255, 255, 255, opacity), font=font)

                    # Composite the overlay onto the frame
                    frame = Image.alpha_composite(frame.convert("RGBA"), overlay)
                frames.append(frame.convert("RGB"))

    elif effect == "scroll":
        # Scroll features vertically across the screen
        total_scroll_height = car_image.height + len(features) * spacing
        num_frames = int(audio_duration * 60)  # 60 FPS for smooth animation
        scroll_speed = total_scroll_height / num_frames

        for frame_idx in range(num_frames):
            offset = frame_idx * scroll_speed
            frame = car_image.copy()
            draw = ImageDraw.Draw(frame)
            for j, feature in enumerate(features):
                y = -len(features) * spacing + (j * spacing) + offset
                if 0 <= y <= car_image.height:
                    dynamic_font_size = int(18 + frame_idx / num_frames * 5)  # Gradually increase font size
                    try:
                        font = ImageFont.truetype("Arial.ttf", dynamic_font_size)
                    except IOError:
                        font = ImageFont.load_default()

                    # Create a dynamic RGB color
                    fancy_color = (
                        int((frame_idx / num_frames) * 255),
                        int((1 - frame_idx / num_frames) * 255),
                        128
                    )
                    text_x = start_x
                    text_y = y
                    draw.text((text_x, text_y), feature, font=font, fill=fancy_color)
            frames.append(frame.convert("RGB"))

    if effect == "pop":
        visible_features = []
        for feature in features:
            visible_features.append(feature)

            for scale in range(1, 11):  # Gradual pop-out animation
                frame = car_image.copy()
                draw = ImageDraw.Draw(frame)

                for j, visible_feature in enumerate(visible_features):
                    text_x = start_x
                    text_y = start_y + j * spacing

                    if visible_feature == feature:
                        # Scale the font size for the popping feature
                        scaled_font_size = int(18 * scale / 10)
                        scaled_font = ImageFont.truetype("Arial.ttf", size=scaled_font_size)

                        # Dynamic fancy color for the popping text
                        pop_color = get_fancy_color(scale)

                        # Draw shadow for readability
                        shadow_x = text_x + 2
                        shadow_y = text_y + 2
                        draw.text((shadow_x, shadow_y), visible_feature, font=scaled_font, fill="black")

                        # Draw the popping feature with dynamic color
                        draw.text((text_x, text_y), visible_feature, font=scaled_font, fill=pop_color)

                    else:
                        # Static font and white color for previously displayed features
                        draw.text((text_x, text_y), visible_feature, font=font, fill="white")

                frames.append(frame.convert("RGB"))



    if effect == "glow":
        visible_features = []
        for feature in features:
            visible_features.append(feature)

            # Glow effect for the current feature, incrementing intensity gradually
            for intensity in range(0, 255, 25):  # Smooth transition for glow intensity
                frame = car_image.copy()
                draw = ImageDraw.Draw(frame)

                for j, visible_feature in enumerate(visible_features):
                    y = start_y + j * spacing

                    # Calculate dynamic color for the glow (color transition from blue to red)
                    glow_color = (
                        int((intensity / 255) * 255),  # Red increases with intensity
                        0,  # Green stays constant
                        int((1 - (intensity / 255)) * 255)  # Blue decreases with intensity
                    )

                    # Draw the glow shadow first (shifted slightly for the effect)
                    shadow_x = start_x + 2
                    shadow_y = y + 2
                    draw.text((shadow_x, shadow_y), visible_feature, font=font, fill=glow_color)

                    # Create a background box to improve text readability (semi-transparent)
                    box_color = (0, 0, 0, 120)  # Semi-transparent black for box
                    box_shape = [(start_x - 5, y - 5), (start_x + box_width + 5, y + feature_height + 5)]
                    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rounded_rectangle(box_shape, radius=10, fill=box_color)
                    frame = Image.alpha_composite(frame.convert("RGBA"), overlay)

                    # Draw the actual text on top of the glow (with a fancy color)
                    text_color = (255, 255, 255)  # White text for readability
                    draw.text((start_x, y), visible_feature, font=font, fill=text_color)

                frames.append(frame.convert("RGB"))


    elif effect == "slide":
        slide_increment = 30  # Increment for smooth sliding
        hold_frames = 10  # Number of frames to hold the final position
        slide_start_x = -500  # Starting position for sliding (off-screen)
        max_slide_x = start_x  # Target final x position

        for slide_x in range(slide_start_x, max_slide_x + 1, slide_increment):  # Smooth sliding effect
            frame = car_image.copy()
            draw = ImageDraw.Draw(frame)

            for j, feature in enumerate(features):
                # Calculate text size to determine dynamic box width
                text_bbox = font.getbbox(feature)  # Updated to use getbbox()
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                box_width = text_width + 2 * box_padding  # Add padding for the box
                feature_height = text_height + 2 * box_padding

                # Calculate y-position for each feature
                y = start_y + j * spacing

                # Define the box and draw it sliding horizontally
                box_shape = [(slide_x, y), (slide_x + box_width, y + feature_height)]
                draw.rectangle(box_shape, fill=box_color)

                # Draw the text inside the sliding box
                text_x = slide_x + box_padding
                text_y = y + box_padding
                draw.text((text_x, text_y), feature, font=font, fill=text_color)

            # Add the frame to the animation
            frames.append(frame.convert("RGB"))

        # Add frames to hold the final position
        for _ in range(hold_frames):
            frame = car_image.copy()
            draw = ImageDraw.Draw(frame)

            for j, feature in enumerate(features):
                # Calculate text size to determine dynamic box width
                text_bbox = font.getbbox(feature)  # Updated to use getbbox()
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                box_width = text_width + 2 * box_padding  # Add padding for the box
                feature_height = text_height + 2 * box_padding

                # Calculate y-position for each feature
                y = start_y + j * spacing

                # Define the box and draw it at the final position
                box_shape = [(max_slide_x, y), (max_slide_x + box_width, y + feature_height)]
                draw.rectangle(box_shape, fill=box_color)

                # Draw the text inside the box
                text_x = max_slide_x + box_padding
                text_y = y + box_padding
                draw.text((text_x, text_y), feature, font=font, fill=text_color)

            # Add the frame to the animation
            frames.append(frame.convert("RGB"))


            
    # Typewriter Effect
    elif effect == "typewriter":
        for feature in features:
            for char_idx in range(1, len(feature) + 1):
                frame = car_image.copy()
                draw = ImageDraw.Draw(frame)
                visible_text = feature[:char_idx]

                y = start_y + features.index(feature) * spacing
                box_shape = [(start_x, y), (start_x + box_width, y + feature_height)]
                draw.rectangle(box_shape, fill=box_color)

                text_x = start_x + box_padding
                text_y = y + box_padding
                draw.text((text_x, text_y), visible_text, font=font, fill=text_color)

                # Append frame to the animation
                frames.append(frame.convert("RGB"))

    # Bounce Effect
    elif effect == "bounce":
        bounce_height = 50  # Maximum bounce height
        bounce_duration = 15  # Frames for a complete bounce cycle
        for frame_idx in range(len(features) * bounce_duration):
            frame = car_image.copy()
            draw = ImageDraw.Draw(frame)

            for j, feature in enumerate(features):
                y_base = start_y + j * spacing
                bounce_offset = bounce_height * abs(
                    math.sin(2 * math.pi * (frame_idx - j * bounce_duration / len(features)) / bounce_duration)
                )
                y = y_base - bounce_offset

                box_shape = [(start_x, y), (start_x + box_width, y + feature_height)]
                draw.rectangle(box_shape, fill=box_color)

                text_x = start_x + box_padding
                text_y = y + box_padding
                draw.text((text_x, text_y), feature, font=font, fill=text_color)

            frames.append(frame.convert("RGB"))

    # Wave Effect
    elif effect == "wave":
        wave_amplitude = 10  # Height of the wave
        wave_frequency = 2  # Number of wave cycles
        for frame_idx in range(int(audio_duration * 30)):  # 30 FPS
            frame = car_image.copy()
            draw = ImageDraw.Draw(frame)

            for j, feature in enumerate(features):
                y_base = start_y + j * spacing
                wave_offset = wave_amplitude * math.sin(
                    2 * math.pi * (wave_frequency * frame_idx / (audio_duration * 30) + j / len(features))
                )
                y = y_base + wave_offset

                box_shape = [(start_x, y), (start_x + box_width, y + feature_height)]
                draw.rectangle(box_shape, fill=box_color)

                text_x = start_x + box_padding
                text_y = y + box_padding
                draw.text((text_x, text_y), feature, font=font, fill=text_color)

            frames.append(frame.convert("RGB"))
            

    # Implement other effects like "pop", "glow", "slide", "typewriter", etc. following the same pattern

    # Ensure at least one frame is generated
    if not frames:
        frames.append(car_image.convert("RGB"))

    # Calculate frame duration and save the animation
    frame_duration = audio_duration / len(frames)  # Frame duration in seconds
    gif = frames[0]
    gif.save(output_path, save_all=True, append_images=frames[1:], duration=int(frame_duration * 1000), loop=0)

    return output_path


# Function to handle generated image with text effects
def get_generated_image(scene_no, scene,audio_duration):
    """
    Generate an image with a description and apply text effects.

    Parameters:
    - scene_no: Scene number to process.
    - audio_duration: Duration of the audio associated with the scene.
    """
    features = scene["Text on Video"]
    text_effect = scene["text_effect"]
    img_path = f"./images/image{scene_no}.png"
    prompt = scene["Scene Description"]
    generate_image(prompt, img_path)  # Assuming a generate_image function is available
    gif_output_path = f"./temp/scene_{scene_no}.gif"
    create_feature_animation(img_path, features, gif_output_path, effect=text_effect, audio_duration=audio_duration)
    return gif_output_path


