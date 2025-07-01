import numpy as np
import cv2
import textwrap


def append_text_to_image(image: np.ndarray, text: str, font_size=1.0, font_line=cv2.LINE_AA,
                         background=(255,255,255), font_color=(0,0,0)):
    r""" Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)
    blank_image[:,:] = background
    linetype = font_line if font_line is not None else cv2.LINE_8

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y % 2 == 1 : y += 1
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            font_color,
            font_thickness,
            lineType=linetype,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((text_image, image), axis=0)
    return final