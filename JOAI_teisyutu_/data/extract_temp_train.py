import csv
import re
import pandas as pd

def extract_color_info(caption):
    # List of common colors to detect
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'dark']
    
    # Find all colors in the caption
    found_colors = []
    color_dict = {}
    
    # Initialize all colors to 0 (not present)
    for color in colors:
        color_dict[f'has_{color}'] = 0
    
    for color in colors:
        # Look for whole word matches of colors
        if re.search(r'\b' + color + r'\b', caption.lower()):
            found_colors.append(color)
            color_dict[f'has_{color}'] = 1
    
    # Determine primary color (first one found, if any)
    primary_color = found_colors[0] if found_colors else None
    
    # Add original info and one-hot encoded colors
    result = {
        'has_color': len(found_colors) > 0,
        'colors': found_colors,
        'color_count': len(found_colors),
        'primary_color': primary_color
    }
    
    # Add the one-hot encoded colors
    result.update(color_dict)
    
    return result

def extract_temperature_info(caption):
    # Extract temperature range
    temp_range = None
    temp_pattern = r'(\d+(?:\.\d+)?)[°\s]*([CF])\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)[°\s]*([CF])'
    temp_match = re.search(temp_pattern, caption)
    
    # Alternative pattern for ranges in format like "27-28°C"
    alt_temp_pattern = r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)[°\s]*([CF])'
    alt_temp_match = re.search(alt_temp_pattern, caption)
    
    # Simple pattern for detecting any temperature
    simple_temp_pattern = r'(\d+(?:\.\d+)?)[°\s]*([CF])'
    simple_temp_match = re.search(simple_temp_pattern, caption)
    
    # Unit of temperature (F or C)
    temp_unit = None
    temp_min = None
    temp_max = None
    
    if temp_match:
        temp_low = float(temp_match.group(1))
        temp_high = float(temp_match.group(3))
        # Take the unit from the second mention, assuming both are the same
        temp_unit = temp_match.group(4)
        temp_range = (temp_low, temp_high)
        temp_min = temp_low
        temp_max = temp_high
    elif alt_temp_match:
        temp_low = float(alt_temp_match.group(1))
        temp_high = float(alt_temp_match.group(2))
        temp_unit = alt_temp_match.group(3)
        temp_range = (temp_low, temp_high)
        temp_min = temp_low
        temp_max = temp_high
    elif simple_temp_match:
        # Only single temperature found, use it for both low and high
        temp = float(simple_temp_match.group(1))
        temp_unit = simple_temp_match.group(2)
        temp_range = (temp, temp)
        temp_min = temp
        temp_max = temp
    
    # Extract coordinates
    coord_pattern = r'(\d+°\d+\'\d+\"[NS]).*?(\d+°\d+\'\d+\"[EW])'
    coord_match = re.search(coord_pattern, caption)
    
    coordinates = None
    if coord_match:
        coordinates = f"{coord_match.group(1)} {coord_match.group(2)}"
    
    # Check for coordinates
    has_coordinates = bool(re.search(r'\d+°\d+\'\d+\"[NS]', caption) or 
                          re.search(r'coordinates', caption.lower()) or
                          re.search(r'GPS', caption.lower()) or 
                          re.search(r'\d+°\d+\'\d+\"[EW]', caption))
    
    # More specific check for the common coordinate pattern in the dataset
    if "18°32'18\"N 73°43'32\"E" in caption or "18°32'18\"N, 73°43'32\"E" in caption:
        has_coordinates = True
        if not coordinates:
            coordinates = "18°32'18\"N 73°43'32\"E"
    
    return {
        'temp_range': temp_range,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'temp_unit': temp_unit,
        'has_coordinates': has_coordinates,
        'coordinates': coordinates
    }

def main():
    # Read the CSV file
    df = pd.read_csv('data/train_with_folds.csv')
    
    # Extract information from captions
    results = []
    color_results = []
    for caption in df['Caption']:
        info = extract_temperature_info(caption)
        color_info = extract_color_info(caption)
        results.append(info)
        color_results.append(color_info)
    
    # Create a dataframe with the results
    results_df = pd.DataFrame(results)
    color_results_df = pd.DataFrame(color_results)
    
    # Add extracted columns to the original dataframe
    df['temp_min'] = results_df['temp_min']
    df['temp_max'] = results_df['temp_max']
    df['temp_unit'] = results_df['temp_unit']
    df['has_coordinates'] = results_df['has_coordinates']
    df['coordinates'] = results_df['coordinates']
    
    # Add color information
    df['has_color'] = color_results_df['has_color']
    df['colors'] = color_results_df['colors']
    df['color_count'] = color_results_df['color_count']
    df['primary_color'] = color_results_df['primary_color']
    
    # Add one-hot encoded color columns
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'dark']
    for color in colors:
        df[f'has_{color}'] = color_results_df[f'has_{color}']
    
    # Save the enhanced dataframe to a new CSV file
    df.to_csv('data/train_folds_temp.csv', index=False)
    print(f"Created new CSV file: train_folds_temp.csv with {len(df)} rows")
    
    # Calculate statistics
    total_captions = len(results_df)
    
    # Temperature unit statistics
    unit_counts = results_df['temp_unit'].value_counts(dropna=True)
    f_count = unit_counts.get('F', 0)
    c_count = unit_counts.get('C', 0)
    no_unit = results_df['temp_unit'].isna().sum()
    
    # Temperature range statistics
    has_temp_range = results_df['temp_range'].notna().sum()
    no_temp_range = results_df['temp_range'].isna().sum()
    
    # Coordinate statistics
    has_coordinates = results_df['has_coordinates'].sum()
    no_coordinates = total_captions - has_coordinates
    coords_extracted = results_df['coordinates'].notna().sum()
    
    # Color statistics
    has_color = color_results_df['has_color'].sum()
    no_color = total_captions - has_color
    color_counts = color_results_df[color_results_df['has_color']]['primary_color'].value_counts()
    top_colors = color_counts.head(5)
    
    # Calculate temperature range statistics for F and C
    f_ranges = []
    c_ranges = []
    
    for i, row in results_df.iterrows():
        if row['temp_range'] is not None and row['temp_unit'] == 'F':
            f_ranges.append(row['temp_range'])
        elif row['temp_range'] is not None and row['temp_unit'] == 'C':
            c_ranges.append(row['temp_range'])
    
    # Calculate min and max temperatures for F and C
    if f_ranges:
        f_min = min([r[0] for r in f_ranges])
        f_max = max([r[1] for r in f_ranges])
        f_range = (f_min, f_max)
    else:
        f_range = None
    
    if c_ranges:
        c_min = min([r[0] for r in c_ranges])
        c_max = max([r[1] for r in c_ranges])
        c_range = (c_min, c_max)
    else:
        c_range = None
    
    # Print results
    print("\nTemperature Information from Captions:")
    print(f"Total captions analyzed: {total_captions}")
    print("\nTemperature Unit:")
    print(f"Fahrenheit (F): {f_count} ({f_count/total_captions*100:.1f}%)")
    print(f"Celsius (C): {c_count} ({c_count/total_captions*100:.1f}%)")
    print(f"No unit specified: {no_unit} ({no_unit/total_captions*100:.1f}%)")
    
    print("\nTemperature Range:")
    print(f"Captions with temperature range: {has_temp_range} ({has_temp_range/total_captions*100:.1f}%)")
    print(f"Captions without temperature range: {no_temp_range} ({no_temp_range/total_captions*100:.1f}%)")
    
    if f_range:
        print(f"\nFahrenheit temperature range across all captions: {f_range[0]:.1f}°F to {f_range[1]:.1f}°F")
    
    if c_range:
        print(f"Celsius temperature range across all captions: {c_range[0]:.1f}°C to {c_range[1]:.1f}°C")
    
    print("\nCoordinates:")
    print(f"Captions with coordinates: {has_coordinates} ({has_coordinates/total_captions*100:.1f}%)")
    print(f"Captions with extracted coordinates: {coords_extracted} ({coords_extracted/total_captions*100:.1f}%)")
    print(f"Captions without coordinates: {no_coordinates} ({no_coordinates/total_captions*100:.1f}%)")
    
    print("\nColor Information:")
    print(f"Captions with color mentions: {has_color} ({has_color/total_captions*100:.1f}%)")
    print(f"Captions without color mentions: {no_color} ({no_color/total_captions*100:.1f}%)")
    print("\nTop 5 most frequent colors:")
    for color, count in top_colors.items():
        print(f"  {color}: {count} ({count/has_color*100:.1f}% of captions with colors)")

if __name__ == "__main__":
    main()
