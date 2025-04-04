# Color scheme definitions
# Color schemes for ESG dashboard

# Minimalist black and white theme
MONOCHROME_THEME = {
    'background': '#111111',
    'card_background': '#1c1c1c',
    'text': '#f0f0f0',
    'primary_color': '#ffffff',
    'secondary_color': '#cccccc',
    'tertiary_color': '#aaaaaa',
    'quaternary_color': '#888888',
    'border_color': '#333333'
}

# Color scales
QUALITY_SCORE_COLOR_SCALE = [
    '#666666',  # Low quality
    '#888888',
    '#aaaaaa',
    '#cccccc',
    '#ffffff'   # High quality
]

# Reversed color scale for heatmaps
MISSING_DATA_COLOR_SCALE = [
    '#ffffff',  # Complete data
    '#aaaaaa',
    '#666666',
    '#333333'   # Missing data
]

# Impact colors
IMPACT_COLORS = {
    'HIGH': '#ff4444',
    'MEDIUM': '#ff9900',
    'LOW': '#ffdd00'
}

# Get color based on quality score
def get_quality_color(score):
    """
    Get color from quality score
    """
    if score >= 90:
        return MONOCHROME_THEME['primary_color']
    elif score >= 80:
        return MONOCHROME_THEME['secondary_color']
    elif score >= 70:
        return MONOCHROME_THEME['tertiary_color']
    elif score >= 60:
        return MONOCHROME_THEME['quaternary_color']
    else:
        return '#666666'

# Get color based on impact
def get_impact_color(impact):
    """
    Get color from impact level
    """
    return IMPACT_COLORS.get(impact, '#666666')