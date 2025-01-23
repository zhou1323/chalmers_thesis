def has_intersection_between_code_lines(section1, section2):
    """
    Check if two code sections have intersection.
    """
    return max(section1[0], section2[0]) <= min(section1[1], section2[1])
