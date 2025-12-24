import requests
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import zipfile
import io
from shapely.geometry import Point
import math

#helpers 
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


#INDIVIDUAL  SCRAPER FUNCTIONS
def get_division(state_fips):
  return STATE_TO_DIVISION[state_fips]

def get_MSA_status(state_fips, county_fips):
  url = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
  cbsa_crosswalk = pd.read_excel(url, dtype=str,header=2)
  val = cbsa_crosswalk[(cbsa_crosswalk["FIPS State Code"] == state_fips) & (cbsa_crosswalk["FIPS County Code"] == county_fips)]
  return val["CBSA Code"].tolist()[0], val["Metropolitan/Micropolitan Statistical Area"].tolist()[0]

# Configuration: URL of shapefile ZIP
UAC20_URL = "https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20.zip"

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

urban_gdf = load_urban_areas_gdf()

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


#GROUP SCRAPER FUNCTIONS
def load_blockgroups_for_state(state_fips: str):
    """
    Load all block groups for a state from TIGER/Line (2023) directly into a DataFrame.
    Returns columns: geoid, lat, lon
    """
    tiger_url = f"https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_{state_fips}_bg.zip"
    gdf = gpd.read_file(tiger_url)

    # Compute centroid coordinates
    gdf['INTPTLAT'] = gdf.geometry.centroid.y
    gdf['INTPTLON'] = gdf.geometry.centroid.x

    # Extract components from full GEOID
    gdf["state_fips"] = gdf["GEOID"].str[0:2]
    gdf["county_fips"] = gdf["GEOID"].str[2:5]
    gdf["tract"] = gdf["GEOID"].str[5:11]      # 6-digit tract
    gdf["block_group"] = gdf["GEOID"].str[11:]  # 1 digit

    df = pd.DataFrame({
        "geoid": gdf["GEOID"],
        "state_fips": gdf["state_fips"],
        "county_fips": gdf["county_fips"],
        "tract": gdf["tract"],
        "block_group": gdf["block_group"],
        "lat": gdf["INTPTLAT"],
        "lon": gdf["INTPTLON"]
    })

    return df

def haversine(lat1, lon1, lat2, lon2):
    """Return distance in meters between two lat/lon points."""
    R = 6371000  # radius Earth in meters
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def blockgroups_within_radius(state_fips: str, origin_lat: float, origin_lon: float,
                              radius_meters: float):
    """
    Returns a DataFrame of block groups in a state whose centroid is within radius (meters)
    from (origin_lat, origin_lon).
    """
    bg = load_blockgroups_for_state(state_fips)


    # quick bounding-box filter to speed up
    deg = radius_meters / 111_000  # rough conversion meters → degrees
    rough = bg[
        (bg.lat >= origin_lat - deg) &
        (bg.lat <= origin_lat + deg) &
        (bg.lon >= origin_lon - deg) &
        (bg.lon <= origin_lon + deg)
    ].copy()

    rough["dist_m"] = haversine(origin_lat, origin_lon, rough.lat.values, rough.lon.values)
    return rough[rough["dist_m"] <= radius_meters].reset_index(drop=True)

def rough_bgs_from_address(school_address, radius_miles):
  lat, lon = geocode_address_census(school_address)
  radius = radius_miles * 1609.34
  state_fips, county_fips, tract, block_group = get_fips_from_coords(lat,lon)

  rough_bg_df = blockgroups_within_radius(state_fips, lat, lon, radius)
  return rough_bg_df

def add_division_column(df, state_col="state_fips", division_col="CENSUS_D"):
    """
    Adds a new column to df based on STATE_TO_DIVISION mapping.

    Parameters:
    - df: DataFrame containing a column of state FIPS codes
    - state_col: name of the column with state FIPS codes (default 'state_fips')
    - division_col: name of the new column to create (default 'division')
    """
    df[division_col] = df[state_col].map(STATE_TO_DIVISION)
    return df

# Load crosswalk once
cbsa_url = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
cbsa_crosswalk = pd.read_excel(cbsa_url, dtype=str, header=2)

# Keep only necessary columns
cbsa_crosswalk = cbsa_crosswalk[["FIPS State Code", "FIPS County Code", "CBSA Code", "Metropolitan/Micropolitan Statistical Area"]]

def get_MSA_status_vectorized(bg_df):

    merged = pd.merge(bg_df, cbsa_crosswalk, left_on=['state_fips', 'county_fips'], right_on=["FIPS State Code", "FIPS County Code"], how='inner')

    merged = merged.drop(columns=["FIPS State Code", "FIPS County Code"])
    return merged

def classify_urban_vectorized(bg_df, gdf=urban_gdf):
    """
    Vectorized version of classify_urban.
    Inputs:
        lat_series: pd.Series of latitudes
        lon_series: pd.Series of longitudes
        gdf: GeoDataFrame of urban areas (already loaded)
    Returns:
        urban_rural_series: pd.Series with "Urban" or "Rural"
        urban_name_series: pd.Series with urban area name or None
    """
    lat_series = bg_df["lat"]
    lon_series = bg_df["lon"]
    # Create GeoDataFrame of points
    points_gdf = gpd.GeoDataFrame(
        pd.DataFrame({"lat": lat_series, "lon": lon_series}),
        geometry=[Point(xy) for xy in zip(lon_series, lat_series)],
        crs="EPSG:4326"
    )

    # Spatial join with urban areas
    joined = gpd.sjoin(points_gdf, gdf[["NAME20", "geometry"]], how="left", predicate="within")

    # Urban/Rural determination
    bg_df["Urban_Name"] = joined["NAME20"]
    bg_df["URBRUR"] = bg_df["Urban_Name"].apply(lambda x: 1 if pd.notnull(x) else 0)





    return bg_df

def get_driving_matrix(origins_df, dest_lat, dest_lon):
    """
    origins_df: DataFrame with columns ["lat", "lon"] for each origin
    dest_lat, dest_lon: single destination coordinates
    Returns: DataFrame with 'distance_mi' and 'duration_min' columns
    """
    # Prepare coordinates: ORS wants [[lon, lat], ...]
    coordinates = origins_df[['lon', 'lat']].values.tolist()
    coordinates.append([dest_lon, dest_lat])  # destination as last coordinate

    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    body = {
        "locations": coordinates,
        "sources": list(range(len(origins_df))),  # all origins
        "destinations": [len(coordinates) - 1],    # single destination
        "metrics": ["distance", "duration"]
    }

    resp = requests.post(url, json=body, headers=headers).json()

    # Extract distances (meters) and durations (seconds)
    distances = [row[0] * 0.000621371 if row[0] != None else None for row in resp["distances"]]  # convert to miles
    #durations = [t[0] / 60 for t in resp["durations"]]           # convert to minutes
    origins_df["SCHOOL_DISTANCE"] = distances

    origins_df = origins_df.dropna(subset=['SCHOOL_DISTANCE'])


    origins_df["LOG_DIST"] = np.log(origins_df["SCHOOL_DISTANCE"])

    

    return origins_df

def get_census_block_groups_vector(df, api_key=None):




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


    grouped = df.groupby(['state_fips', 'county_fips', 'tract'])
    collected_data = pd.DataFrame(columns=all_vars + ["geoid"])

    for (state, county, tract), group in grouped:
      print("starting next block group")

      bg_list = ",".join(group["block_group"].astype(str))

      params_bg = {
          "get": var_str,
          "for": f"block group:{bg_list}",
          "in": f"state:{state}+county:{county}+tract:{tract}"
      }
      if api_key:
          params_bg["key"] = api_key

      print(base_bg)
      print(params_bg)
      response_bg = requests.get(base_bg, params=params_bg)
      response_bg.raise_for_status()
      data_bg = pd.DataFrame(response_bg.json())

      print("ending next block group")

      data_bg.columns = data_bg.iloc[0]  # set first row as header
      data_bg = data_bg[1:]              # drop the first row
      data_bg = data_bg.reset_index(drop=True)  # optional, reset index

      data_bg["geoid"] = data_bg["state"] + data_bg["county"] + data_bg["tract"] + data_bg["block group"]

      collected_data = pd.concat([collected_data, data_bg], ignore_index=True)
   
    # --- 1. Initialize HH columns ---
    race_cols = [
        "HH_RACE_White",
        "HH_RACE_Black or African American",
        "HH_RACE_Asian",
        "HH_RACE_American Indian or Alaska Native",
        "HH_RACE_Native Hawaiian or other Pacific Islander",
        "HH_RACE_Other",
        "HH_RACE_No data"
    ]

    hisp_cols = ["HH_HISP", "HH_HISP_No data"]

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

    human_cols = ["Block Group Mode Race", "Block Group Mode Hispanic", "Block Group Median Income"]

    for col in race_cols + hisp_cols + income_cols + human_cols:
        collected_data[col] = 0 if col.startswith("HH") else "No data"


    # --- 2. RACE ---
    race_vals = ["B02001_002E", "B02001_003E", "B02001_004E",
                 "B02001_005E", "B02001_006E","HH_RACE_Other_count"]

    # Other race = sum of B02001_007E + B02001_008E
    collected_data["HH_RACE_Other_count"] = collected_data["B02001_007E"].fillna(0).astype(int) + collected_data["B02001_008E"].fillna(0).astype(int)

    # Fill NAs with 0 for safety
    collected_data[race_vals] = collected_data[race_vals].fillna(0).astype(int)

    # Dominant race per row
    dominant_race_idx = collected_data[race_vals].idxmax(axis=1)
    dominant_race_vals = collected_data[race_vals].max(axis=1)

    # One-hot encoding
    for col in race_cols:
      if col != "HH_RACE_No data":
        collected_data[col] = (dominant_race_idx == race_vals[race_cols.index(col)]).astype(int)

    # Mapping from ACS variable to human-readable label (remove 'HH_RACE_')
    race_mapping = {k: v[8:] for k, v in zip(race_vals, race_cols)}

    # --- 5. Human-readable column ---
    collected_data["Block Group Mode Race"] = dominant_race_idx.map(race_mapping)

    # Handle "No data"
    no_data_mask = dominant_race_vals <= 0
    collected_data["HH_RACE_No data"] = no_data_mask.astype(int)
    collected_data.loc[no_data_mask, "Block Group Mode Race"] = "No data"

    # --- 3. HISPANIC ---
    collected_data["HH_HISP"] = (collected_data["B03003_003E"].fillna(0) > collected_data["B03003_002E"].fillna(0)).astype(int)
    collected_data["HH_HISP_No data"] = ((collected_data["B03003_003E"].fillna(0) + collected_data["B03003_002E"].fillna(0)) == 0).astype(int)
    collected_data["Block Group Mode Hispanic"] = np.where(
        collected_data["HH_HISP_No data"] == 1,
        "No data",
        np.where(collected_data["HH_HISP"] == 1, "Hispanic", "Not Hispanic")
    )

    # --- 4. INCOME --
    income_bins = [
        (0, 9999, 'HHFAMINC_Less than $10,000'),
        (10000, 14999, 'HHFAMINC_$10,000 to $14,999'),
        (15000, 24999, 'HHFAMINC_$15,000 to $24,999'),
        (25000, 34999, 'HHFAMINC_$25,000 to $34,999'),
        (35000, 49999, 'HHFAMINC_$35,000 to $49,999'),
        (50000, 74999, 'HHFAMINC_$50,000 to $74,999'),
        (75000, 99999, 'HHFAMINC_$75,000 to $99,999'),
        (100000, 124999, 'HHFAMINC_$100,000 to $124,999'),
        (125000, 149999, 'HHFAMINC_$125,000 to $149,999'),
        (150000, 199999, 'HHFAMINC_$150,000 to $199,999'),
        (200000, np.inf, 'HHFAMINC_$200,000 or more')
    ]

    collected_data["Block Group Median Income"] = collected_data["B19013_001E"]
    collected_data["B19013_001E"] = pd.to_numeric(collected_data["B19013_001E"], errors='coerce')

    for lower, upper, col_name in income_bins:
        mask = collected_data["B19013_001E"].between(lower, upper)
        collected_data.loc[mask, col_name] = 1

    # Handle missing income
    missing_mask = collected_data["B19013_001E"] < 0
    collected_data.loc[missing_mask, "HHFAMINC_No data"] = 1
    collected_data.loc[missing_mask, "Block Group Median Income"] = "No data"

    # --- 5. Drop original ACS variables if desired ---
    acs_vars = ["B19013_001E", "B02001_002E","B02001_003E","B02001_004E",
                "B02001_005E","B02001_006E","B02001_007E","B02001_008E",
                "B03003_002E","B03003_003E","HH_RACE_Other_count",'state_fips', 'county_fips', 'tract', 'block group']
    collected_data.drop(columns=acs_vars, inplace=True, errors='ignore')

    # --- 6. Merge back with original df on geoid ---
    final_df = df.merge(collected_data, on="geoid", how="left")



    return final_df

def add_msa_size_columns(df, api_key='5056c54df9531a543b539bc29a812a083bff0e92'):
    """
    Adds MSA size one-hot columns and a human-readable 'Home MSA Size' column.

    df must contain:
        - 'state_fips'
        - 'county_fips'
        - 'CBSA Code'
        - 'Metropolitan/Micropolitan Statistical Area'
        - 'msa_status' (optional; used to mark Micropolitan)

    Returns df with:
        - 'cbsa_population'
        - MSA size one-hot columns
        - 'Home MSA Size'
    """

    # --- Step 1: Identify CBSAs to fetch (exclude micropolitan) ---
    cbsa_to_fetch = df.loc[
        ~df["Metropolitan/Micropolitan Statistical Area"].str.contains("Micropolitan", na=False),
        "CBSA Code"
    ].dropna().unique()

    df["Home MSA Size"] = "Not in MSA or CMSA"
    msa_cols = [
        'MSASIZE_In an MSA of Less than 250,000',
        'MSASIZE_In an MSA of 250,000 - 499,999',
        'MSASIZE_In an MSA of 500,000 - 999,999',
        'MSASIZE_In an MSA or CMSA of 1,000,000 - 2,999,999',
        'MSASIZE_In an MSA or CMSA of 3 million or more',
    ]
    for col in msa_cols:
        df[col] = 0
    df['MSASIZE_Not in MSA or CMSA'] = 1

    #nun to fetch
    if len(cbsa_to_fetch) == 0:
        return df

    # --- Step 2: Fetch CBSA populations ---
    base_cbsa = "https://api.census.gov/data/2022/acs/acs5"
    cbsa_list_str = ",".join(cbsa_to_fetch.astype(str))
    params = {
        "get": "B01003_001E",
        "for": f"metropolitan statistical area/micropolitan statistical area:{cbsa_list_str}"
    }
    if api_key:
        params["key"] = api_key

    resp = requests.get(base_cbsa, params=params)
    resp.raise_for_status()
    data = resp.json()

    pop_df = pd.DataFrame(data[1:], columns=data[0])
    pop_df["B01003_001E"] = pop_df["B01003_001E"].astype(int)

    pop_df.set_index("metropolitan statistical area/micropolitan statistical area", inplace=True)

    # --- Step 3: Map populations back to original df ---
    df["cbsa_population"] = df["CBSA Code"].map(pop_df["B01003_001E"])

    # Optional: mark micropolitan / non-MSA counties as None
    df.loc[df["Metropolitan/Micropolitan Statistical Area"].str.contains("Micropolitan", na=False), "cbsa_population"] = None

    # --- Step 4: Add MSA size one-hot columns ---
    msa_cols = [
        'MSASIZE_In an MSA of Less than 250,000',
        'MSASIZE_In an MSA of 250,000 - 499,999',
        'MSASIZE_In an MSA of 500,000 - 999,999',
        'MSASIZE_In an MSA or CMSA of 1,000,000 - 2,999,999',
        'MSASIZE_In an MSA or CMSA of 3 million or more',
        'MSASIZE_Not in MSA or CMSA'
    ]
    for col in msa_cols:
        df[col] = 0

    # Human-readable default
    df["Home MSA Size"] = "Not in MSA or CMSA"

    # --- Step 5: Fill one-hot and human-readable values ---
    for idx, pop in df["cbsa_population"].items():
        if pd.isna(pop):
            # Not in MSA or Micropolitan
            df.at[idx, 'MSASIZE_Not in MSA or CMSA'] = 1
            df.at[idx, 'Home MSA Size'] = "Not in MSA or CMSA"
        else:
            if pop < 250_000:
                key = 'MSASIZE_In an MSA of Less than 250,000'
            elif pop <= 499_999:
                key = 'MSASIZE_In an MSA of 250,000 - 499,999'
            elif pop <= 999_999:
                key = 'MSASIZE_In an MSA of 500,000 - 999,999'
            elif pop <= 2_999_999:
                key = 'MSASIZE_In an MSA or CMSA of 1,000,000 - 2,999,999'
            else:
                key = 'MSASIZE_In an MSA or CMSA of 3 million or more'
            df.at[idx, key] = 1
            df.at[idx, 'Home MSA Size'] = pop


    df = df.drop("cbsa_population", axis=1)
    return df

def add_tract_land_area(df, api_key='5056c54df9531a543b539bc29a812a083bff0e92'):
    """
    df: DataFrame with columns ['state_fips', 'county_fips', 'tract']
    Returns df with an additional column:
        - land_area_sqmi
    """
    base = "https://api.census.gov/data/2023/geoinfo"
    results = []

    # Group by state+county to batch tracts
    for (state, county), group in df.groupby(["state_fips", "county_fips"]):
        tracts_list = group["tract"].astype(str).tolist()
        tract_query = ",".join(tracts_list)

        params = {
            "get": "AREALAND_SQMI",
            "for": f"tract:{tract_query}",
            "in": f"state:{state}+county:{county}"
        }
        if api_key:
            params["key"] = api_key

        resp = requests.get(base, params=params)
        resp.raise_for_status()
        data = resp.json()

        # First row = column names
        cols = data[0]
        for row in data[1:]:
            row_dict = dict(zip(cols, row))
            results.append({
                "state_fips": state,
                "county_fips": county,
                "tract": row_dict["tract"],
                "land_area_sqmi": float(row_dict["AREALAND_SQMI"])
            })

    # Create a dataframe of results
    land_df = pd.DataFrame(results)

    # Merge back with the original df
    df = df.merge(land_df, on=["state_fips", "county_fips", "tract"], how="left")

    return df

def get_census_tract_data_vectorized(df, api_key='5056c54df9531a543b539bc29a812a083bff0e92'):
    """
    df: DataFrame with ['state_fips', 'county_fips', 'tract', 'lat', 'lon', 'land_area_sqmi']
    Returns df with additional vectorized columns for tract densities, percent renter, and one-hot bins.
    Assumes land_area_sqmi already exists.
    """

    # --- 1. Fetch tract-level ACS data ---
    base = "https://api.census.gov/data/2022/acs/acs5"
    tract_vars = [
        "B23025_003E", "B23025_006E",   # Employed male, employed female
        "B25003_002E", "B25003_003E",   # Owner occ, renter occ
        "B01003_001E",                  # Total population
        "B25001_001E"                   # Housing units
    ]
    var_str = ",".join(tract_vars)

    results = []

    for (state, county), group in df.groupby(["state_fips", "county_fips"]):
        tracts_list = group["tract"].astype(str).tolist()
        tract_query = ",".join(tracts_list)

        params = {
            "get": var_str,
            "for": f"tract:{tract_query}",
            "in": f"state:{state}+county:{county}"
        }
        if api_key:
            params["key"] = api_key

        resp = requests.get(base, params=params)
        resp.raise_for_status()
        data = resp.json()

        cols = data[0]
        for row in data[1:]:
            row_dict = dict(zip(cols, row))
            emp_male = int(row_dict["B23025_003E"])
            emp_female = int(row_dict["B23025_006E"])
            owner_occ = int(row_dict["B25003_002E"])
            renter_occ = int(row_dict["B25003_003E"])
            pop_total = int(row_dict["B01003_001E"])
            housing_units = int(row_dict["B25001_001E"])
            results.append({
                "state_fips": state,
                "county_fips": county,
                "tract": row_dict["tract"],
                "tract_workers": emp_male + emp_female,
                "percent_renter_occupied": renter_occ / (owner_occ + renter_occ) * 100 if (owner_occ + renter_occ) > 0 else None,
                "tract_population": pop_total,
                "tract_housing_units": housing_units
            })

    tract_data = pd.DataFrame(results)


    df = df.merge(tract_data, on=["state_fips", "county_fips", "tract"], how="left")

    # --- 2. Compute densities using existing land area ---
    df["pdensity"] = df["tract_population"] / df["land_area_sqmi"]
    df["wdensity"] = df["tract_workers"] / df["land_area_sqmi"]
    df["hdensity"] = df["tract_housing_units"] / df["land_area_sqmi"]

    # --- 3. One-hot bins for percent renter ---
    def renter_bin(x):
        if x < 5: return 0
        elif x < 15: return 5
        elif x >= 95: return 95
        else: return (x + 5) // 10 * 10

    df["OTHTNRNT"] = df["percent_renter_occupied"].apply(renter_bin)

    # --- 4. One-hot bins for population density ---
    def pop_bin(x):
        if x < 100: return 50
        elif x < 500: return 300
        elif x < 1000: return 750
        elif x < 2000: return 1500
        elif x < 4000: return 3000
        elif x < 10000: return 7000
        elif x < 25000: return 17000
        else: return 30000

    df["OTPPOPDN"] = df["pdensity"].apply(pop_bin)

    # --- 5. One-hot bins for worker density ---
    def worker_bin(x):
        if x < 50: return 25
        elif x < 100: return 75
        elif x < 250: return 150
        elif x < 500: return 350
        elif x < 1000: return 750
        elif x < 2000: return 1500
        elif x < 4000: return 3000
        else: return 5000

    df["OTEEMPDN"] = df["wdensity"].apply(worker_bin)

    # --- 6. One-hot bins for housing unit density ---
    df["OTRESDN"] = df["hdensity"].apply(pop_bin)



    return df

def add_school_tract_features(state_fips, county_fips, tract, tract_df, api_key='5056c54df9531a543b539bc29a812a083bff0e92'):
    """
    Adds DTHTNRNT, DTPPOPDN, DTEEMPDN, DTRESDN to df.
    All rows get the same values, based on the given tract.

    Parameters:
    - df: DataFrame to modify
    - state_fips, county_fips, tract: tract identifiers
    - tract_df: precomputed tract dataframe with OTHTNRNT, OTPPOPDN, OTEEMPDN, OTRESDN, land_area_sqmi
    - api_key: optional Census API key

    Returns:
    - df with the four new columns
    """

    # --- 1. Try to find the tract in precomputed data ---
    match = tract_df[
        (tract_df["state_fips"] == state_fips) &
        (tract_df["county_fips"] == county_fips) &
        (tract_df["tract"] == tract)
    ]

    if not match.empty:
        row = match.iloc[0]
        values = {
            "DTHTNRNT": row["OTHTNRNT"],
            "DTPPOPDN": row["OTPPOPDN"],
            "DTEEMPDN": row["OTEEMPDN"],
            "DTRESDN": row["OTRESDN"]
        }
    else:
        # --- 2. If not found, fetch via Census API ---
        base = "https://api.census.gov/data/2022/acs/acs5"
        tract_vars = [
            "B23025_003E", "B23025_006E",   # Employed male, employed female
            "B25003_002E", "B25003_003E",   # Owner occ, renter occ
            "B01003_001E",                  # Total population
            "B25001_001E"                   # Housing units
        ]
        var_str = ",".join(tract_vars)

        params = {
            "get": var_str,
            "for": f"tract:{tract}",
            "in": f"state:{state_fips}+county:{county_fips}"
        }
        if api_key:
            params["key"] = api_key

        resp = requests.get(base, params=params)
        resp.raise_for_status()
        data = resp.json()
        row_data = dict(zip(data[0], data[1]))

        emp_male = int(row_data["B23025_003E"])
        emp_female = int(row_data["B23025_006E"])
        owner_occ = int(row_data["B25003_002E"])
        renter_occ = int(row_data["B25003_003E"])
        pop_total = int(row_data["B01003_001E"])
        housing_units = int(row_data["B25001_001E"])

        # Land area: try from tract_df if exists
        land_area = match["land_area_sqmi"].iloc[0] if "land_area_sqmi" in match.columns and not match.empty else 1.0

        pdensity = pop_total / land_area
        wdensity = (emp_male + emp_female) / land_area
        hdensity = housing_units / land_area
        percent_renter_occupied = renter_occ / (owner_occ + renter_occ) * 100 if (owner_occ + renter_occ) > 0 else 0

        # Binning functions
        def renter_bin(x):
            if x < 5: return 0
            elif x < 15: return 5
            elif x >= 95: return 95
            else: return (x + 5) // 10 * 10

        def pop_bin(x):
            if x < 100: return 50
            elif x < 500: return 300
            elif x < 1000: return 750
            elif x < 2000: return 1500
            elif x < 4000: return 3000
            elif x < 10000: return 7000
            elif x < 25000: return 17000
            else: return 30000

        def worker_bin(x):
            if x < 50: return 25
            elif x < 100: return 75
            elif x < 250: return 150
            elif x < 500: return 350
            elif x < 1000: return 750
            elif x < 2000: return 1500
            elif x < 4000: return 3000
            else: return 5000

        values = {
            "DTHTNRNT": renter_bin(percent_renter_occupied),
            "DTPPOPDN": pop_bin(pdensity),
            "DTEEMPDN": worker_bin(wdensity),
            "DTRESDN": pop_bin(hdensity)
        }

    # --- 3. Add to all rows in df ---
    for col, val in values.items():
        tract_df[col] = val

    return tract_df

def matrix_generator(school_address, radius):
  lat,lon = geocode_address_census(school_address)
  state_fips, county_fips, tract, bg = get_fips_from_coords(lat,lon)
  print("School FIPS Completed")
  block_groups = rough_bgs_from_address(school_address, radius)
  print("Block Groups Completed")
  block_groups = get_MSA_status_vectorized(block_groups)
  print("MSA Status Completed")
  block_groups = classify_urban_vectorized(block_groups)
  print("Urban Status Completed")
  block_groups = get_driving_matrix(block_groups, lat, lon)
  print("Driving Distance Completed")
  block_groups = get_census_block_groups_vector(block_groups)
  print("Census BG Data Completed")
  block_groups = add_msa_size_columns(block_groups)
  print("MSA Size Completed")
  block_groups = add_tract_land_area(block_groups)
  print("Land Area Completed")
  block_groups = get_census_tract_data_vectorized(block_groups)
  print("Tract Data Completed")
  block_groups = add_school_tract_features(state_fips, county_fips, tract, block_groups)
  print("Destination Data Completed")
  block_groups = add_division_column(block_groups)
  print("CENSUS_D Completed")

  return block_groups

from typing import Iterable, Optional
import warnings

def df_to_geojson(
    df: pd.DataFrame,
    geoid_col: str = "geoid",
    lat_col: str = "lat",
    lon_col: str = "lon",
    tiger_year: int = 2023,
    bg_prefix_template: str = "https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{st}_bg.zip",
    props_cols: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Convert a pandas DataFrame with block-group identifiers into a GeoJSON FeatureCollection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a block-group identifier column (geoid_col). Recommended: also lat/lon.
    geoid_col : str
        Column name containing the block-group GEOID (state+county+tract+blockgroup).
    lat_col, lon_col : str
        Column names for fallback point geometry when no polygon found.
    tiger_year : int
        TIGER year to use for BG URLs (default 2023).
    bg_prefix_template : str
        URL template for downloading statewide BG zip (must contain {year} and {st}).
    props_cols : iterable or None
        If provided, only include these columns from df in feature properties. Otherwise include all df columns.
    verbose : bool
        If True prints progress/debug info.

    Returns
    -------
    geojson_str : str
        GeoJSON FeatureCollection (string) ready to send to frontend.
    """

    if geoid_col not in df.columns:
        raise ValueError(f"geoid column '{geoid_col}' not found in dataframe")

    df_work = df.copy()

    # Normalize geoid to string and zero-pad to 12 (block group GEOID length)
    df_work[geoid_col] = df_work[geoid_col].astype(str).str.zfill(12)

    # optional selection of properties
    if props_cols is None:
        prop_columns = [c for c in df_work.columns if c != geoid_col]
    else:
        prop_columns = list(props_cols)
        # ensure geoid is always included as property on features
        if geoid_col not in prop_columns:
            prop_columns = prop_columns + [geoid_col]

    # Determine states to download
    df_work["__state_fips__"] = df_work[geoid_col].str[:2]
    states = sorted(df_work["__state_fips__"].dropna().unique())

    if verbose:
        print("States detected:", states)

    # Start with an empty GeoDataFrame for merged results
    merged_gdfs = []

    # For each state, attempt to download the statewide BG shapefile and match
    for st in states:
        st = str(st).zfill(2)
        url = bg_prefix_template.format(year=tiger_year, st=st)
        if verbose:
            print(f"Attempting to download TIGER BG for state {st} -> {url}")

        try:
            # Read statewide BG shapefile. GeoPandas can read zip urls directly.
            gdf_state = gpd.read_file(url)
        except Exception as e:
            warnings.warn(f"Failed to load TIGER BG for state {st}: {e}. Rows for this state will fallback to points.")
            gdf_state = None

        # Subset dataframe rows for this state
        rows_state = df_work[df_work["__state_fips__"] == st].copy()
        if rows_state.empty:
            continue

        if gdf_state is not None and "GEOID" in gdf_state.columns:
            # Ensure CRS is WGS84
            try:
                gdf_state = gdf_state.to_crs(epsg=4326)
            except Exception:
                # if already in 4326 or fails, continue, but ensure geometry exists
                pass

            # Keep only the GEOIDs we need (speed)
            want_geoids = rows_state[geoid_col].unique().tolist()
            gdf_subset = gdf_state[gdf_state["GEOID"].astype(str).isin(want_geoids)].copy()

            # normalize geoid column name for merging
            gdf_subset["geoid"] = gdf_subset["GEOID"].astype(str).str.zfill(12)

            # Merge polygon geometries into rows_state
            rows_state = rows_state.merge(
                gdf_subset[["geoid", "geometry"]],
                left_on=geoid_col, right_on="geoid", how="left"
            )
            # geometry column may be named 'geometry' already from merge
            if "geometry" in rows_state.columns:
                gstate = gpd.GeoDataFrame(rows_state, geometry="geometry", crs="EPSG:4326")
            else:
                gstate = gpd.GeoDataFrame(rows_state, geometry=None, crs="EPSG:4326")
        else:
            # No shapefile available -> create GeoDataFrame without geometry for now
            gstate = gpd.GeoDataFrame(rows_state, geometry=None, crs="EPSG:4326")

        merged_gdfs.append(gstate)

    # Concatenate all states
    if merged_gdfs:
        all_gdf = pd.concat(merged_gdfs, ignore_index=True)
    else:
        # nothing matched; create GeoDataFrame from original df
        all_gdf = gpd.GeoDataFrame(df_work.copy(), geometry=None, crs="EPSG:4326")

    # If geometry missing for rows, attempt to build from lat/lon
    if ("geometry" not in all_gdf.columns) or all_gdf.geometry.isnull().any():
        # ensure lat/lon exist
        if lat_col in all_gdf.columns and lon_col in all_gdf.columns:
            # create geometry where missing
            missing_mask = all_gdf.geometry.isnull() if "geometry" in all_gdf.columns else pd.Series(True, index=all_gdf.index)
            pts = [Point(xy) if not pd.isna(xy[0]) and not pd.isna(xy[1]) else None
                   for xy in zip(all_gdf[lon_col], all_gdf[lat_col])]
            # assign points
            all_gdf.loc[missing_mask, "geometry"] = pd.Series(
                [p for i, p in enumerate(pts) if missing_mask.iat[i]],
                index=all_gdf.index[missing_mask]
            )
        else:
            # no lat/lon to build fallback; geometry stays NaN
            warnings.warn("No lat/lon columns present to build fallback Point geometry for missing polygons.")

    # Ensure we have a GeoDataFrame and CRS
    all_gdf = gpd.GeoDataFrame(all_gdf, geometry="geometry", crs="EPSG:4326")

    # If any rows still lack geometry, drop them (or keep but they won't plot); here we keep but log
    still_missing = all_gdf.geometry.isnull().sum()
    if still_missing > 0:
        warnings.warn(f"{still_missing} rows have no geometry even after fallback. They will be omitted from GeoJSON geometries.")

    # Prepare properties: include requested prop columns (and geoid)
    props_final = [c for c in prop_columns if c in all_gdf.columns]
    # always include geoid column in properties
    if geoid_col not in props_final:
        props_final = [geoid_col] + props_final

    # Build a GeoDataFrame containing only geometry + props_final in a consistent order
    out_gdf = all_gdf[props_final + ["geometry"]].copy()

    # Convert numeric types that are numpy types to native python to make JSON clean (optional)
    # Use GeoPandas' to_json
    geojson_str = out_gdf.to_json()  # FeatureCollection string

    if verbose:
        print("GeoJSON features:", len(out_gdf.dropna(subset=["geometry"])))

    return geojson_str

