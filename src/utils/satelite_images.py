import requests
import os
import random
import math
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')
OUTPUT_DIR = './data/raw'
SIZE = '500x500'
ZOOM = 19          # High detail for building segmentation
MAPTYPE = 'satellite'
IMAGES_PER_CITY = 10
RADIUS_KM = 4.0 

# List of top 50 Polish cities with approximate center coordinates (Latitude, Longitude)
CITIES = [
    ("Warszawa", 52.2297, 21.0122), ("Krakow", 50.0647, 19.9450), ("Lodz", 51.7592, 19.4560),
    ("Wroclaw", 51.1079, 17.0385), ("Poznan", 52.4064, 16.9252), ("Gdansk", 54.3520, 18.6466),
    ("Szczecin", 53.4285, 14.5528), ("Bydgoszcz", 53.1235, 18.0084), ("Lublin", 51.2465, 22.5684),
    ("Bialystok", 53.1325, 23.1688), ("Katowice", 50.2649, 19.0238), ("Gdynia", 54.5189, 18.5305),
    ("Czestochowa", 50.8118, 19.1203), ("Radom", 51.4027, 21.1471), ("Torun", 53.0138, 18.5984),
    ("Sosnowiec", 50.2863, 19.1041), ("Rzeszow", 50.0412, 21.9991), ("Kielce", 50.8661, 20.6286),
    ("Gliwice", 50.2945, 18.6714), ("Zabrze", 50.3249, 18.7857), ("Olsztyn", 53.7784, 20.4801),
    ("Bielsko-Biala", 49.8225, 19.0444), ("Bytom", 50.3480, 18.9328), ("Zielona_Gora", 51.9356, 15.5062),
    ("Rybnik", 50.0971, 18.5418), ("Ruda_Slaska", 50.2643, 18.8633), ("Opole", 50.6751, 17.9213),
    ("Tychy", 50.1259, 18.9890), ("Gorzow_Wlkp", 52.7325, 15.2369), ("Elblag", 54.1561, 19.4045),
    ("Plock", 52.5463, 19.7065), ("Dabrowa_Gornicza", 50.3216, 19.1864), ("Walbrzych", 50.7818, 16.2691),
    ("Wloclawek", 52.6482, 19.0678), ("Tarnow", 50.0121, 20.9858), ("Chorzow", 50.2976, 18.9546),
    ("Koszalin", 54.1944, 16.1722), ("Kalisz", 51.7673, 18.0853), ("Legnica", 51.2070, 16.1551),
    ("Grudziadz", 53.4839, 18.7535), ("Jaworzno", 50.2033, 19.2749), ("Slupsk", 54.4641, 17.0285),
    ("Jastrzebie_Zdroj", 49.9577, 18.5833), ("Nowy_Sacz", 49.6218, 20.6970), ("Jelenia_Gora", 50.9044, 15.7194),
    ("Siedlce", 52.1677, 22.2902), ("Myslowice", 50.2373, 19.1432), ("Konin", 52.2235, 18.2512),
    ("Pila", 53.1511, 16.7378), ("Piotrkow_Tryb", 51.4052, 19.7032)
]

def get_random_location(lat, lon, radius_km):
    """
    Generates a random point within a radius_km circle around (lat, lon).
    Uses uniform distribution logic to avoid clustering in the center.
    """
    # 1 degree of latitude is ~111.132 km
    # 1 degree of longitude is ~111.320*cos(lat) km
    
    r = radius_km * math.sqrt(random.random()) # sqrt ensures uniform area distribution
    theta = random.random() * 2 * math.pi
    
    # Calculate offsets in km
    dy = r * math.cos(theta)
    dx = r * math.sin(theta)
    
    # Convert offsets to degrees
    new_lat = lat + (dy / 111.132)
    # Cosine requires radians
    new_lon = lon + (dx / (111.320 * math.cos(math.radians(lat))))
    
    return new_lat, new_lon

# --- Main Execution ---

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Target: 50 cities, {IMAGES_PER_CITY} images each. Total: {len(CITIES) * IMAGES_PER_CITY}")

total_downloaded = 0

for city_name, city_lat, city_lon in CITIES:
    print(f"--- Processing {city_name} ---")
    
    for i in range(IMAGES_PER_CITY):
        # Generate random coordinates
        lat, lon = get_random_location(city_lat, city_lon, RADIUS_KM)
        
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={ZOOM}&size={SIZE}&maptype={MAPTYPE}&key={API_KEY}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Naming convention: city_index_lat_lon.png
                filename = f"{city_name}_{i+1:02d}_{lat:.4f}_{lon:.4f}.png"
                file_path = os.path.join(OUTPUT_DIR, filename)
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                total_downloaded += 1
            else:
                print(f"Failed to download image for {city_name}: {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching {city_name}: {e}")

print(f"\nCompleted! Total images downloaded: {total_downloaded}")
print(f"Images saved in directory: {OUTPUT_DIR}")