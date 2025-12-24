import requests
import pandas as pd
import geopandas as gpd
import requests
import zipfile
import io
from shapely.geometry import Point
import math

def geocode_address_census(address):
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"

    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "format": "json"
    }

    response = requests.get(url, params=params).json()
    matches = response["result"]["addressMatches"]

    if len(matches) == 0:
        raise ValueError("Address not found.")

    coords = matches[0]["coordinates"]
    return coords["y"], coords["x"]   # (lat, lon)

def get_fips_from_coords(lat, lon):
    """
    Given lat/lon, return:
    - state FIPS
    - county FIPS
    - tract code
    - block group code
    """
    url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"
    response = requests.get(url).json()

    block_fips = response["Block"]["FIPS"]  # 15-digit block code

    state_fips = block_fips[:2]       # 2 digits
    county_fips = block_fips[2:5]     # 3 digits
    tract = block_fips[5:11]          # 6 digits
    block = block_fips[11:]           # 4 digits
    block_group = block[0]            # 1 digit

    return state_fips, county_fips, tract, block_group

STATE_TO_DIVISION = {
    # New England
    "09":"New England","23":"New England","25":"New England","33":"New England","44":"New England","50":"New England",

    # Middle Atlantic
    "34":"Middle Atlantic","36":"Middle Atlantic","42":"Middle Atlantic",

    # East North Central
    "17":"East North Central","18":"East North Central","26":"East North Central","39":"East North Central","55":"East North Central",

    # West North Central
    "19":"West North Central","20":"West North Central","27":"West North Central","29":"West North Central",
    "31":"West North Central","38":"West North Central","46":"West North Central",

    # South Atlantic
    "10":"South Atlantic","11":"South Atlantic","12":"South Atlantic","13":"South Atlantic",
    "24":"South Atlantic","37":"South Atlantic","45":"South Atlantic","51":"South Atlantic","54":"South Atlantic",

    # East South Central
    "01":"East South Central","21":"East South Central","28":"East South Central","47":"East South Central",

    # West South Central
    "05":"West South Central","22":"West South Central","40":"West South Central","48":"West South Central",

    # Mountain
    "04":"Mountain","08":"Mountain","16":"Mountain","30":"Mountain",
    "32":"Mountain","35":"Mountain","49":"Mountain","56":"Mountain",

    # Pacific
    "02":"Pacific","06":"Pacific","15":"Pacific","41":"Pacific","53":"Pacific"
}

def get_division(state_fips):
  return STATE_TO_DIVISION[state_fips]


def get_MSA_status(state_fips, county_fips):
  url = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
  cbsa_crosswalk = pd.read_excel(url, dtype=str,header=2)
  val = cbsa_crosswalk[(cbsa_crosswalk["FIPS State Code"] == state_fips) & (cbsa_crosswalk["FIPS County Code"] == county_fips)]
  return val["CBSA Code"].tolist()[0], val["Metropolitan/Micropolitan Statistical Area"].tolist()[0]


# ----------------------------------------------------
# Configuration: URL of shapefile ZIP
UAC20_URL = "https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20.zip"

# ----------------------------------------------------
# Helper to load shapefile into GeoDataFrame
def load_urban_areas_gdf(url=UAC20_URL):
    # Download zip into bytes
    r = requests.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Find the .shp file name inside the zip
    shapefile_name = [f for f in z.namelist() if f.endswith(".shp")][0]

    # Extract all files into memory buffer
    z.extractall("/tmp/tl_uac20")

    # Load with GeoPandas
    gdf = gpd.read_file(f"/tmp/tl_uac20/{shapefile_name}")
    # Ensure it's in WGS84 lat/lon
    gdf = gdf.to_crs(epsg=4326)
    return gdf

# Load once
urban_gdf = load_urban_areas_gdf()

# ----------------------------------------------------
def classify_urban(lat, lon, gdf=urban_gdf):
    """
    Returns:
      - 'Urban' if the point is inside any urban polygon
      - 'Rural' otherwise
      - urban area name if inside urban, else None
    """
    pt = Point(lon, lat)
    match = gdf[gdf.contains(pt)]
    if not match.empty:
        # Inside some urban polygon
        name = match.iloc[0]["NAME20"]
        return "Urban", name
    else:
        return "Rural", None  # Not in urban area
    

# Get a free key from https://openrouteservice.org/sign-up/
API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImFiMTU3YmFjMzYxNzQ3MGRhZGY5ZWQ4MTFmOTE0ZGZiIiwiaCI6Im11cm11cjY0In0="

def get_driving_distance_ors(address1, address2):
    """
    Returns driving distance in kilometers and duration in minutes using OpenRouteService.
    """
    # First, geocode addresses using ORS
    def geocode(address):
        url = "https://api.openrouteservice.org/geocode/search"
        params = {"api_key": API_KEY, "text": address, "size": 1}
        resp = requests.get(url, params=params).json()
        if len(resp["features"]) == 0:
            raise ValueError(f"Address not found: {address}")
        coords = resp["features"][0]["geometry"]["coordinates"]  # [lon, lat]
        return coords

    start_coords = geocode(address1)
    end_coords = geocode(address2)

    # Call directions endpoint
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    body = {
        "coordinates": [start_coords, end_coords]
    }
    resp = requests.post(url, json=body, headers=headers).json()
    route = resp["routes"][0]["summary"]
    distance_mi = route["distance"] / 1000 * 0.621371
    duration_min = route["duration"] / 60

    return distance_mi, duration_min

def get_census_block_group_data(state_fips, county_fips, tract, block_group, api_key=None):
    """
    Fetch block group median income, race/Hispanic counts
    in one function call.

    Parameters:
        state_fips (str): 2-digit state FIPS
        county_fips (str): 3-digit county FIPS
        tract (str): 6-digit tract code
        block_group (str): 1-digit block group code
        api_key (str, optional): Your Census API key

    Returns:
        dict: {
            "median_income": int or "No data",
            "race_counts": dict of race/Hispanic counts,
        }
    """
    # ---------- Block group: median income + race/Hispanic ----------
    base_bg = "https://api.census.gov/data/2022/acs/acs5"

    race_vars = [
        "B02001_002E",  # White
        "B02001_003E",  # Black or African American
        "B02001_004E",  # Asian
        "B02001_005E",  # American Indian or Alaska Native
        "B02001_006E",  # Native Hawaiian or other Pacific Islander
        "B02001_007E",  # Other race
        "B02001_008E",  # Two or more races
    ]
    hisp_vars = [
        "B03003_002E",  # Not Hispanic
        "B03003_003E"   # Hispanic
    ]

    all_vars = ["B19013_001E"] + race_vars + hisp_vars
    var_str = ",".join(all_vars)

    params_bg = {
        "get": var_str,
        "for": f"block group:{block_group}",
        "in": f"state:{state_fips}+county:{county_fips}+tract:{tract}"
    }
    if api_key:
        params_bg["key"] = api_key

    response_bg = requests.get(base_bg, params=params_bg)
    response_bg.raise_for_status()
    data_bg = response_bg.json()

    # Median income
    median_income_raw = data_bg[1][0]
    median_income = int(median_income_raw) if median_income_raw not in [None, "", "null"] else "No data"

    # Race/Hispanic counts
    counts_raw = dict(zip(data_bg[0][1:], data_bg[1][1:]))  # skip median income key
    race_counts = {k: int(v) for k, v in counts_raw.items()}


    # ---------- Combine results ----------
    return {
        "median_income": median_income,
        "race_counts": race_counts,
    }

def get_census_CBSA_data(state_fips, county_fips, api_key=None):

    # ---------- Block group: median income + race/Hispanic ----------
    base_bg = "https://api.census.gov/data/2022/acs/acs5"

    # ---------- CBSA population ----------
    cbsa_code, msa_status = get_MSA_status(state_fips, county_fips)
    if "Micropolitan" in msa_status:
      return "Not in MSA/CMSA"

    base_cbsa = "https://api.census.gov/data/2022/acs/acs5"
    params_cbsa = {
        "get": "B01003_001E",
        "for": f"metropolitan statistical area/micropolitan statistical area:{cbsa_code}"
    }
    if api_key:
        params_cbsa["key"] = api_key

    response_cbsa = requests.get(base_cbsa, params=params_cbsa)
    response_cbsa.raise_for_status()
    data_cbsa = response_cbsa.json()
    cbsa_population = int(data_cbsa[1][0])


    return cbsa_population


def get_census_tract_data(state_fips, county_fips, tract, api_key=None):
    """
    Return:
      - workers_per_sq_mile
      - pct_renter
      - population_density
      - housing_units_per_sq_mile
      - tract_land_area_sqmi (filled later)
    """

    base = "https://api.census.gov/data/2022/acs/acs5"

    tract_vars = [
        "B23025_003E", "B23025_006E",   # Employed male, employed female
        "B25003_002E", "B25003_003E",   # Owner occ, renter occ
        "B01003_001E",                  # Total population
        "B25001_001E"                   # Housing units
    ]

    params_tract = {
        "get": ",".join(tract_vars),
        "for": f"tract:{tract}",
        "in": f"state:{state_fips}+county:{county_fips}"
    }
    if api_key:
        params_tract["key"] = api_key

    tract_data = requests.get(base, params=params_tract).json()
    trow = tract_data[1]

    emp_male  = int(trow[0])
    emp_female = int(trow[1])
    owner_occ  = int(trow[2])
    renter_occ = int(trow[3])
    pop_total  = int(trow[4])
    housing_units = int(trow[5])


    return {
        "tract_workers": emp_male + emp_female,
        "percent_renter_occupied": renter_occ/(owner_occ+renter_occ),
        "tract_population": pop_total,
        "tract_housing_units": housing_units,
    }


def get_tract_land_area(state_fips, county_fips, tract, api_key=None):
    """
    Return land area (m² and sq mi) for the given census tract
    using GEOINFO 2023 dataset.
    """
    base = "https://api.census.gov/data/2023/geoinfo"

    params = {
        "get": "AREALAND,AREALAND_SQMI",
        "for": f"tract:{tract}",
        "in": f"state:{state_fips}+county:{county_fips}"
    }
    if api_key:
        params["key"] = api_key

    resp = requests.get(base, params=params)
    resp.raise_for_status()
    data = resp.json()

    if len(data) < 2:
        raise ValueError("No geography data returned for tract")

    land_m2 = float(data[1][0])
    land_sqmi = float(data[1][1])

    return {"land_area_m2": land_m2, "land_area_sqmi": land_sqmi}


def vector_generator(home_address, school_address):
  final_vector = {}
  hv = {}

  lat, lon = geocode_address_census(home_address)
  state_fips, county_fips, tract, block_group = get_fips_from_coords(lat,lon)
  hv["Home Latitude"], hv["Home Longitude"], hv["Home State FIPS Code"], hv["Home County FIPS Code"], hv["Home Tract Number"], hv["Home Block Group Number"] = lat, lon, state_fips, county_fips, tract, block_group
  
  slat, slon = geocode_address_census(school_address)
  sstate_fips, scounty_fips, stract, sblock_group = get_fips_from_coords(slat,slon)
  hv["School Latitude"], hv["School Longitude"], hv["School State FIPS Code"], hv["School County FIPS Code"], hv["School Tract Number"], hv["School Block Group Number"] = slat, slon, sstate_fips, scounty_fips, stract, sblock_group
  

  #school distance
  mi, min = get_driving_distance_ors(home_address, school_address)
  
  final_vector["LOG_DIST"] = math.log(mi)
  hv["Distance to School"] = mi

  #census division
  final_vector["CENSUS_D"] = STATE_TO_DIVISION[state_fips]
  hv["Home Census Division"] = final_vector["CENSUS_D"]

  #CBSA code
  hv["Home CBSA Code"] = get_MSA_status(state_fips, county_fips)[0]

  #census data

  #race, hisp, and median income
  census_data = get_census_block_group_data(state_fips, county_fips, tract, block_group)

  # --- Race data ---
  race_counts = census_data["race_counts"]

  # Extract individual race counts (default to 0 if missing)
  white = race_counts.get("B02001_002E", 0)
  black = race_counts.get("B02001_003E", 0)
  asian = race_counts.get("B02001_004E", 0)
  native_american = race_counts.get("B02001_005E", 0)
  pacific_islander = race_counts.get("B02001_006E", 0)
  other = race_counts.get("B02001_007E", 0) + race_counts.get("B02001_008E", 0)

  # --- One-hot column names to actual human-readable labels ---
  race_label_map = {
      "HH_RACE_White": "White",
      "HH_RACE_Black or African American": "Black or African American",
      "HH_RACE_Asian": "Asian",
      "HH_RACE_American Indian or Alaska Native": "American Indian or Alaska Native",
      "HH_RACE_Native Hawaiian or other Pacific Islander": "Native Hawaiian or other Pacific Islander",
      "HH_RACE_Other": "Other",
      "HH_RACE_No data": "No data"
  }

  # If everything is missing → mark No Data
  if sum([white, black, asian, native_american, pacific_islander, other]) == 0:
      for key in race_label_map:
          final_vector[key] = 1 if key == "HH_RACE_No data" else 0

      hv["Estimated Race"] = "No data"

  else:
      # Determine dominant race by max count
      race_mapping = {
          "HH_RACE_White": white,
          "HH_RACE_Black or African American": black,
          "HH_RACE_Asian": asian,
          "HH_RACE_American Indian or Alaska Native": native_american,
          "HH_RACE_Native Hawaiian or other Pacific Islander": pacific_islander,
          "HH_RACE_Other": other
      }

      top_race_key = max(race_mapping, key=race_mapping.get)
      top_race_label = race_label_map[top_race_key]

      # One-hot encoding
      for key in race_label_map:
          final_vector[key] = 1 if key == top_race_key else 0

      # Assign to human readable vector
      hv["Block Group Mode Race"] = top_race_label

  #hisp data
  # 003E = Hispanic, 002E = Not Hispanic
  hisp_counts = race_counts
  if hisp_counts.get("B03003_003E", 0) > hisp_counts.get("B03003_002E", 0):
      final_vector["HH_HISP"] = 1
      final_vector["HH_HISP_No data"] = 0
      hv["Block Group Mode Hispanic"] = "Hispanic"
  elif hisp_counts.get("B03003_003E", 0) <= hisp_counts.get("B03003_002E", 0):
      final_vector["HH_HISP"] = 0
      final_vector["HH_HISP_No data"] = 0
      hv["Block Group Mode Hispanic"] = "Not Hispanic"
  else:
      final_vector["HH_HISP_No data"] = 1
      hv["Block Group Mode Hispanic"] = "No data"
    

  # --- Median Household Income (one-hot) ---
  median_income = census_data.get("median_income")

  income_cols = [
      'HHFAMINC_Less than $10,000',
      'HHFAMINC_$10,000 to $14,999',
      'HHFAMINC_$15,000 to $24,999',
      'HHFAMINC_$25,000 to $34,999',
      'HHFAMINC_$35,000 to $49,999',
      'HHFAMINC_$50,000 to $74,999',
      'HHFAMINC_$75,000 to $99,999',
      'HHFAMINC_$100,000 to $124,999',
      'HHFAMINC_$125,000 to $149,999',
      'HHFAMINC_$150,000 to $199,999',
      'HHFAMINC_$200,000 or more',
      'HHFAMINC_No data'
  ]

  # initialize all income one-hot cols to 0
  for col in income_cols:
      final_vector[col] = 0

  # default human-readable
  hv["Estimated Income"] = "No data"

  if isinstance(median_income, int):
      if median_income < 10000:
          bucket = 'HHFAMINC_Less than $10,000'
      elif median_income <= 14999:
          bucket = 'HHFAMINC_$10,000 to $14,999'
  
      elif median_income <= 24999:
          bucket = 'HHFAMINC_$15,000 to $24,999'

      elif median_income <= 34999:
          bucket = 'HHFAMINC_$25,000 to $34,999'

      elif median_income <= 49999:
          bucket = 'HHFAMINC_$35,000 to $49,999'

      elif median_income <= 74999:
          bucket = 'HHFAMINC_$50,000 to $74,999'

      elif median_income <= 99999:
          bucket = 'HHFAMINC_$75,000 to $99,999'

      elif median_income <= 124999:
          bucket = 'HHFAMINC_$100,000 to $124,999'
    
      elif median_income <= 149999:
          bucket = 'HHFAMINC_$125,000 to $149,999'

      elif median_income <= 199999:
          bucket = 'HHFAMINC_$150,000 to $199,999'
        
      else:
          bucket = 'HHFAMINC_$200,000 or more'


      # set the one-hot and human-readable/string fields
      if bucket in income_cols:
          final_vector[bucket] = 1
      hv["Estimated Income"] = bucket[9:]
  else:
      # leave all income one-hots as 0 and human as No data
      final_vector["HHFAMINC_No data"] = 1



  # --- MSA Size (one-hot) ---
  cbsa_population = get_census_CBSA_data(state_fips, county_fips)

  msa_cols = [
      'MSASIZE_In an MSA of Less than 250,000',
      'MSASIZE_In an MSA of 250,000 - 499,999',
      'MSASIZE_In an MSA of 500,000 - 999,999',
      'MSASIZE_In an MSA or CMSA of 1,000,000 - 2,999,999',
      'MSASIZE_In an MSA or CMSA of 3 million or more',
      'MSASIZE_Not in MSA or CMSA'
  ]

  # initialize
  for col in msa_cols:
      final_vector[col] = 0

  # default text fields
  hv["Home MSA Size"] = "Not in MSA or CMSA"

  if isinstance(cbsa_population, int):
      if cbsa_population < 250000:
          msa_key = 'MSASIZE_In an MSA of Less than 250,000'

      elif cbsa_population <= 499999:
          msa_key = 'MSASIZE_In an MSA of 250,000 - 499,999'

      elif cbsa_population <= 999999:
          msa_key = 'MSASIZE_In an MSA of 500,000 - 999,999'

      elif cbsa_population <= 2999999:
          msa_key = 'MSASIZE_In an MSA or CMSA of 1,000,000 - 2,999,999'
        
      else:
          msa_key = 'MSASIZE_In an MSA or CMSA of 3 million or more'
      

      final_vector[msa_key] = 1
      hv["Home MSA Size"] = msa_key[8:]
  else:
      # If cbsa_population is None or indicates micropolitan / not in MSA
      final_vector['MSASIZE_Not in MSA or CMSA'] = 1
      hv["Home MSA Size"] = "Not in MSA or CMSA"


  # --- Urban / Rural (keep existing behavior and human_vector) ---
  urban_rural, name = classify_urban(lat, lon)
  final_vector["URBRUR"] = 1 if urban_rural == "Urban" else 0
  hv["Home Urban/Rural"] = urban_rural

  #Origin Data
  tract_area = get_tract_land_area(state_fips, county_fips, tract)["land_area_sqmi"]
  tract_data = get_census_tract_data(state_fips, county_fips, tract)
  wdensity = tract_data["tract_workers"]/tract_area
  percent_renter_occupied = tract_data["percent_renter_occupied"] * 100
  pdensity = tract_data["tract_population"]/tract_area
  hdensity = tract_data["tract_housing_units"]/tract_area
  
  #origin renter occupied housing
  if percent_renter_occupied < 5:
    final_vector["OTHTNRNT"] = 0
  elif percent_renter_occupied < 15:
    final_vector["OTHTNRNT"] = 5
  elif percent_renter_occupied >= 95:
    final_vector["OTHTNRNT"] = 95
  else:
    final_vector["OTHTNRNT"] = (percent_renter_occupied+5) //10 * 10
  hv["Home Tract % Renter Occupied Housing"] = percent_renter_occupied

  #origin population density

  if pdensity < 100:
      final_vector["OTPPOPDN"] = 50
  elif pdensity < 500:
      final_vector["OTPPOPDN"] = 300
  elif pdensity < 1000:
      final_vector["OTPPOPDN"] = 750
  elif pdensity < 2000:
      final_vector["OTPPOPDN"] = 1500
  elif pdensity < 4000:
      final_vector["OTPPOPDN"] = 3000
  elif pdensity < 10000:
      final_vector["OTPPOPDN"] = 7000
  elif pdensity < 25000:
      final_vector["OTPPOPDN"] = 17000
  else:
      final_vector["OTPPOPDN"] = 30000

  hv["Home Tract Population Density (per sq mile)"] = pdensity

  #origin worker density

  if wdensity < 50:
      final_vector["OTEEMPDN"] = 25
  elif wdensity < 100:
      final_vector["OTEEMPDN"] = 75
  elif wdensity < 250:
      final_vector["OTEEMPDN"] = 150
  elif wdensity < 500:
      final_vector["OTEEMPDN"] = 350
  elif wdensity < 1000:
      final_vector["OTEEMPDN"] = 750
  elif wdensity < 2000:
      final_vector["OTEEMPDN"] = 1500
  elif wdensity < 4000:
      final_vector["OTEEMPDN"] = 3000
  else:
      final_vector["OTEEMPDN"] = 5000

  hv["Home Tract Workers per sq mile"] = wdensity
  
  #housing unit density
  
  if hdensity < 100:
      final_vector["OTRESDN"] = 50
  elif hdensity < 500:
      final_vector["OTRESDN"] = 300
  elif hdensity < 1000:
      final_vector["OTRESDN"] = 750
  elif hdensity < 2000:
      final_vector["OTRESDN"] = 1500
  elif hdensity < 4000:
      final_vector["OTRESDN"] = 3000
  elif hdensity < 10000:
      final_vector["OTRESDN"] = 7000
  elif hdensity < 25000:
      final_vector["OTRESDN"] = 17000
  else:
      final_vector["OTRESDN"] = 30000

  hv["Home Tract Housing Units per sq mile"] = hdensity

  #Destination data
  if sstate_fips == state_fips and stract == tract:
    hv["School Tract % Renter Occupied Housing"] = hv["Home Tract % Renter Occupied Housing"]
    hv["School Tract Population Density (per sq mile)"] = hv["Home Tract Population Density (per sq mile)"]
    hv["School Tract Workers per sq mile"] = hv["Home Tract Workers per sq mile"] 
    hv["School Tract Housing Units per sq mile"] = hv["Home Tract Housing Units per sq mile"]

  else:
    tract_area = get_tract_land_area(sstate_fips, scounty_fips, stract)["land_area_sqmi"]
    tract_data = get_census_tract_data(sstate_fips, scounty_fips, stract)

    wdensity = tract_data["tract_workers"] / tract_area
    percent_renter_occupied = tract_data["percent_renter_occupied"] * 100
    pdensity = tract_data["tract_population"] / tract_area
    hdensity = tract_data["tract_housing_units"] / tract_area


    # ---- Destination renter-occupied housing (D THTNRNT) ----
    if percent_renter_occupied < 5:
        final_vector["DTHTNRNT"] = 0
    elif percent_renter_occupied < 15:
        final_vector["DTHTNRNT"] = 5
    elif percent_renter_occupied >= 95:
        final_vector["DTHTNRNT"] = 95
    else:
        final_vector["DTHTNRNT"] = (percent_renter_occupied + 5) // 10 * 10

    hv["School Tract % Renter Occupied Housing"] = percent_renter_occupied


    # ---- Destination population density (D TPPOPDN) ----
    if pdensity < 100:
        final_vector["DTPPOPDN"] = 50
    elif pdensity < 500:
        final_vector["DTPPOPDN"] = 300
    elif pdensity < 1000:
        final_vector["DTPPOPDN"] = 750
    elif pdensity < 2000:
        final_vector["DTPPOPDN"] = 1500
    elif pdensity < 4000:
        final_vector["DTPPOPDN"] = 3000
    elif pdensity < 10000:
        final_vector["DTPPOPDN"] = 7000
    elif pdensity < 25000:
        final_vector["DTPPOPDN"] = 17000
    else:
        final_vector["DTPPOPDN"] = 30000

    hv["School Tract Population Density (per sq mile)"] = pdensity


    # ---- Destination worker density (D TEEMPDN) ----
    if wdensity < 50:
        final_vector["DTEEMPDN"] = 25
    elif wdensity < 100:
        final_vector["DTEEMPDN"] = 75
    elif wdensity < 250:
        final_vector["DTEEMPDN"] = 150
    elif wdensity < 500:
        final_vector["DTEEMPDN"] = 350
    elif wdensity < 1000:
        final_vector["DTEEMPDN"] = 750
    elif wdensity < 2000:
        final_vector["DTEEMPDN"] = 1500
    elif wdensity < 4000:
        final_vector["DTEEMPDN"] = 3000
    else:
        final_vector["DTEEMPDN"] = 5000

    hv["School Tract Workers per sq mile"] = wdensity


    # ---- Destination housing units density (D TRESDN) ----
    if hdensity < 100:
        final_vector["DTRESDN"] = 50
    elif hdensity < 500:
        final_vector["DTRESDN"] = 300
    elif hdensity < 1000:
        final_vector["DTRESDN"] = 750
    elif hdensity < 2000:
        final_vector["DTRESDN"] = 1500
    elif hdensity < 4000:
        final_vector["DTRESDN"] = 3000
    elif hdensity < 10000:
        final_vector["DTRESDN"] = 7000
    elif hdensity < 25000:
        final_vector["DTRESDN"] = 17000
    else:
        final_vector["DTRESDN"] = 30000

    hv["School Tract Housing Units per sq mile"] = hdensity
      

  return final_vector, hv

