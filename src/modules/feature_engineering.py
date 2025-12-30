import collections.abc
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import h3
import logging

logger = logging.getLogger(__name__)

def calc_dist(p1x, p1y, p2x, p2y):
  
  """
  
  """

  p1 = (p2x - p1x)**2
  p2 = (p2y - p1y)**2
  dist = np.sqrt(p1 + p2)

  return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else dist

def get_restaurant_ids(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:

    #TODO This function takes in a dataframe. In the future, maybe modify this so it is consistent with other functions, where df is not part of input
  
    df_copy = df.copy()
    
    #unique restaurants
    restaurants_ids = {}

    for a,b in zip(df_copy.restaurant_lat, df_copy.restaurant_lon):
        id = "{}_{}".format(a,b)
        restaurants_ids[id] = {"lat": a, "lon":b}
    for i,key in enumerate(restaurants_ids.keys()):
        restaurants_ids[key]['id'] = i

    #labeling of restaurants
    df_copy['restaurant_id']=[restaurants_ids["{}_{}".format(a,b)]['id'] for a,b in zip(df_copy.restaurant_lat, df_copy.restaurant_lon)]

    # number of unique restaurants
    return restaurants_ids, df_copy


def avg_dist_to_restaurants(courier_lat:float, courier_lon:float, restaurants_ids: dict) -> float:
  """

  """


  return np.mean([calc_dist(v['lat'], v['lon'], courier_lat, courier_lon) for v in restaurants_ids.values()])

def calc_haversine_dist(lat1, lon1, lat2, lon2):
  """

  """

  R = 6372.8    #3959.87433  this is in miles.  For Earth radius in kilometers use 6372.8 km
  if isinstance(lat1, collections.abc.Sequence):
    dLat = np.array([radians(l2 - l1) for l2,l1 in zip(lat2, lat1)])
    dLon = np.array([radians(l2 - l1) for l2,l1 in zip(lon2, lon1)])
    lat1 = np.array([radians(l) for l in lat1])
    lat2 = np.array([radians(l) for l in lat2])
  else:
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

  a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
  c = 2*np.arcsin(np.sqrt(a))
  dist = R*c

  return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist

def avg_Hdist_to_restaurants(courier_lat:float, courier_lon:float, restaurants_ids: dict) -> float:
  """

  """
  
  return np.mean([calc_haversine_dist(v['lat'], v['lon'], courier_lat, courier_lon) for v in restaurants_ids.values()])

def initiate_centroids(k: int, df: pd.DataFrame) -> pd.DataFrame:
    '''
    Select k data points as centroids
    k: number of centroids
    dset: pandas dataframe
    '''

    df_copy = df.copy()
    centroids = df_copy.sample(k)

    return centroids

def eucl_dist(p1x,p1y,p2x,p2y): #TODO Can remove this function or rename calc_dist function?
  return calc_dist(p1x, p1y, p2x, p2y)

def centroid_assignation(df, centroids):
  """
  
  """
  df_copy = df.copy()
#   k = len(centroids) #TODO not used
#   n = len(df_copy) #TODO not used
  assignation = []
  assign_errors = []
  centroids_list = [c for i,c in centroids.iterrows()]
  
  for i,obs in df_copy.iterrows():
    # Estimate error
    all_errors = [eucl_dist(centroid['lat'],
                            centroid['lon'],
                            obs['courier_lat'],
                            obs['courier_lon']) for centroid in centroids_list]

    # Get the nearest centroid and the error
    nearest_centroid =  np.where(all_errors==np.min(all_errors))[0].tolist()[0]
    nearest_centroid_error = np.min(all_errors)

    # Add values to corresponding lists
    assignation.append(nearest_centroid)
    assign_errors.append(nearest_centroid_error)

  df_copy['Five_Clusters_embedding'] = assignation
  df_copy['Five_Clusters_embedding_error'] = assign_errors
  
  return df_copy

def create_h3_index(df:pd.DataFrame, resolution:int) -> pd.DataFrame:
    """
    
    """

    df_copy = df.copy()

    df_copy['h3_index'] = [h3.geo_to_h3(lat,lon,resolution) for (lat,lon) in zip(df_copy.courier_lat, df_copy.courier_lon)]

    return df_copy

def date_time_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    
    """

    df_copy = df.copy()

    df_copy['order_created_timestamp'] = pd.to_datetime(df_copy['order_created_timestamp'], format='mixed')
    df_copy['courier_location_timestamp']=  pd.to_datetime(df_copy['courier_location_timestamp'], format='mixed')
    df_copy['date_day_number'] = [d for d in df_copy.courier_location_timestamp.dt.day_of_year]
    df_copy['date_hour_number'] = [d for d in df_copy.courier_location_timestamp.dt.hour]

    return df_copy

def orders_busyness(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """

    df_copy = df.copy()

    index_list = [(i,d,hr) for (i,d,hr) in zip(df_copy.h3_index, df_copy.date_day_number, df_copy.date_hour_number)]

    set_indexes = list(set(index_list))
    dict_indexes = {label: index_list.count(label) for label in set_indexes}
    df_copy['orders_busyness_by_h3_hour'] = [dict_indexes[i] for i in index_list]

    return df_copy

def restaurant_index(df:pd.DataFrame) -> pd.DataFrame:
    """
    
    """

    df_copy = df.copy()
    restaurants_counts_per_h3_index = {a:len(b) for a,b in zip(df_copy.groupby('h3_index')['restaurant_id'].unique().index, df_copy.groupby('h3_index')['restaurant_id'].unique()) }
    df_copy['restaurants_per_index'] = [restaurants_counts_per_h3_index[h] for h in df_copy.h3_index]

    return df_copy
