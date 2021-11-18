from palettable.cartocolors import qualitative

def palette(N):
    if N == 2:
        return qualitative.Vivid_2.hex_colors
    elif N == 3:
        return qualitative.Vivid_3.hex_colors
    elif N == 4:
        return qualitative.Vivid_4.hex_colors
    elif N == 5:
        return qualitative.Vivid_5.hex_colors
    elif N == 6:
        return qualitative.Vivid_6.hex_colors
    elif N == 7:
        return qualitative.Vivid_7.hex_colors
    elif N == 8:
        return qualitative.Vivid_8.hex_colors
    elif N == 9:
        return qualitative.Vivid_9.hex_colors
    elif N == 10:
        return qualitative.Vivid_10.hex_colors
    else:
        return 10 * qualitative.Vivid_10.hex_colors

