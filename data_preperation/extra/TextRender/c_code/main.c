#include <glib.h>
#include <pango/pangocairo.h>
#include <string.h>
#include <stdlib.h>

#include "render_text_image.h"


int main (int argc, char ** argv)
{
    if (argc < 9){
        printf("Error: must receive at least 8 parameters - <text to print> <output path> <fonts directory> <font family> <font weight index 0-5> <font stretch 0-8> <letter spacing 0-3> <font size 0-3> <DEBUG(optional)>");
        printf("Received %d parameters!\n", argc - 1);
        return -1;
    }
    char DEBUG = 1;
    const char * const FONT_WEIGHTS[] = { "ultralight", "light", "normal", "bold", "ultrabold", "heavy" };
    const char * const FONT_STRETCHES[] = { "ultracondensed", "extracondensed", "condensed", "semicondensed", \
  "normal", "semiexpanded", "expanded", "extraexpanded", "ultraexpanded" };
    const char * const LETTER_SPACING[] = { "0", "1000", "2000" };
    const char * const FONT_SIZE[] = { "10000", "150000", "20000", "25000" };

    const char *text_to_print = argv[1];
    char *output_path = malloc(sizeof(char) * (strlen(argv[2]) + 1));
    strcpy(output_path, argv[2]);

    char *fonts_directory = argv[3];
    const char *font_family = argv[4];

    const char * font_weight = FONT_WEIGHTS[atoi(argv[5])];
    const char * font_stretch = FONT_STRETCHES[atoi(argv[6])];
    const char * letter_spacing = LETTER_SPACING[atoi(argv[7])];
    const char * font_size = FONT_SIZE[atoi(argv[8])];

    if (argc == 10) {
        DEBUG = atoi(argv[9]);
    }

    if (DEBUG == 1) {
        printf("Using parameters:\nText to print=%s\nOutput path=%s\nFonts directory=%s\nFont family=%s\nFont weight=%s\nFont stretch=%s\nLetter spacing=%s\nFont size=%s\n",
        text_to_print, output_path, fonts_directory, font_family, font_weight, font_stretch, letter_spacing, font_size);
    }

    char *markup_pattern = "<span foreground='black' background='white' font_family='%s' font_size='%s' font_weight='%s' font_stretch='%s' letter_spacing=%s></span>";
    //printf("%s\n",markup_pattern);
    char *markup_text = malloc(sizeof(char) * (strlen(markup_pattern) + strlen(font_weight) + strlen(font_stretch) \
    + strlen(font_family) + strlen(text_to_print) + strlen(letter_spacing) + 10 + strlen(font_size) )) ;
    //printf("%s\n",markup_pattern);
    sprintf(markup_text, "<span foreground='black' background='white' font_family='%s' font_size='%s' font_weight='%s' font_stretch='%s' letter_spacing='%s'>%s</span>",
     font_family, font_size, font_weight, font_stretch, letter_spacing, text_to_print);
    if (DEBUG == 1) {
        printf("Using markup text:\n%s\n", markup_text);
    }
    create_text_image(output_path, markup_text, fonts_directory);
    free(output_path);
    free(markup_text);
    return 0;
     
}
