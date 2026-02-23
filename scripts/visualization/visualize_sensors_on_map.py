"""
OpenStreetMap을 사용하여 센서들을 지도에 표시하는 함수들
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


def load_sensor_metadata(metadata_path: str) -> pd.DataFrame:
    """
    센서 metadata 파일을 로드합니다.
    
    Args:
        metadata_path: metadata 파일 경로 (CSV 형식)
        
    Returns:
        센서 정보가 담긴 DataFrame (sensor_id, latitude, longitude 컬럼 포함)
    """
    df = pd.read_csv(metadata_path)
    
    # 필수 컬럼 확인
    required_cols = ['sensor_id', 'latitude', 'longitude']
    if not all(col in df.columns for col in required_cols):
        # 다른 가능한 컬럼 이름 확인
        possible_id_cols = ['sensor_id', 'sensor_idx', 'station_id', 'node_id', 'id']
        possible_lat_cols = ['latitude', 'lat', 'y']
        possible_lon_cols = ['longitude', 'lon', 'lng', 'x']
        
        id_col = next((col for col in possible_id_cols if col in df.columns), None)
        lat_col = next((col for col in possible_lat_cols if col in df.columns), None)
        lon_col = next((col for col in possible_lon_cols if col in df.columns), None)
        
        if id_col and lat_col and lon_col:
            df = df.rename(columns={id_col: 'sensor_id', lat_col: 'latitude', lon_col: 'longitude'})
        else:
            raise ValueError(f"필수 컬럼을 찾을 수 없습니다. 필요한 컬럼: {required_cols}")
    
    # 좌표 유효성 검사
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
    
    return df


def create_sensor_map(
    sensors_df: pd.DataFrame,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 10,
    marker_size: int = 5,
    show_popup: bool = True
) -> folium.Map:
    """
    OpenStreetMap에 센서들을 표시하는 지도를 생성합니다.
    
    Args:
        sensors_df: 센서 정보 DataFrame (sensor_id, latitude, longitude 컬럼 포함)
        center_lat: 지도 중심 위도 (None이면 센서들의 평균 사용)
        center_lon: 지도 중심 경도 (None이면 센서들의 평균 사용)
        zoom_start: 초기 줌 레벨
        marker_size: 마커 크기
        show_popup: 센서 ID를 팝업으로 표시할지 여부
        
    Returns:
        folium.Map 객체
    """
    # 중심 좌표 계산
    if center_lat is None:
        center_lat = sensors_df['latitude'].mean()
    if center_lon is None:
        center_lon = sensors_df['longitude'].mean()
    
    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # 센서 마커 추가
    for idx, row in sensors_df.iterrows():
        sensor_id = row['sensor_id']
        lat = row['latitude']
        lon = row['longitude']
        
        popup_text = f"Sensor ID: {sensor_id}" if show_popup else None
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_size,
            popup=popup_text,
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.6,
            weight=1
        ).add_to(m)
    
    # 마커 클러스터 추가 (센서가 많을 경우)
    if len(sensors_df) > 100:
        marker_cluster = plugins.MarkerCluster().add_to(m)
        for idx, row in sensors_df.iterrows():
            sensor_id = row['sensor_id']
            lat = row['latitude']
            lon = row['longitude']
            popup_text = f"Sensor ID: {sensor_id}" if show_popup else None
            folium.Marker(
                location=[lat, lon],
                popup=popup_text
            ).add_to(marker_cluster)
    
    return m


def visualize_sensors_from_file(
    metadata_path: str,
    output_path: Optional[str] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 10,
    marker_size: int = 5,
    show_popup: bool = True
) -> folium.Map:
    """
    metadata 파일에서 센서 정보를 로드하고 지도에 시각화합니다.
    
    Args:
        metadata_path: 센서 metadata 파일 경로
        output_path: HTML 파일로 저장할 경로 (None이면 저장하지 않음)
        center_lat: 지도 중심 위도 (None이면 센서들의 평균 사용)
        center_lon: 지도 중심 경도 (None이면 센서들의 평균 사용)
        zoom_start: 초기 줌 레벨
        marker_size: 마커 크기
        show_popup: 센서 ID를 팝업으로 표시할지 여부
        
    Returns:
        folium.Map 객체
    """
    # 센서 정보 로드
    sensors_df = load_sensor_metadata(metadata_path)
    
    print(f"로드된 센서 개수: {len(sensors_df)}")
    print(f"위도 범위: {sensors_df['latitude'].min():.4f} ~ {sensors_df['latitude'].max():.4f}")
    print(f"경도 범위: {sensors_df['longitude'].min():.4f} ~ {sensors_df['longitude'].max():.4f}")
    
    # 지도 생성
    m = create_sensor_map(
        sensors_df=sensors_df,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_start=zoom_start,
        marker_size=marker_size,
        show_popup=show_popup
    )
    
    # HTML 파일로 저장
    if output_path is not None:
        m.save(output_path)
        print(f"지도가 저장되었습니다: {output_path}")
    
    return m


if __name__ == "__main__":
    # 예제 사용법
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python visualize_sensors_on_map.py <metadata_path> [output_path]")
        print("예제: python visualize_sensors_on_map.py datasets/metadata.csv output.html")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "sensor_map.html"
    
    m = visualize_sensors_from_file(
        metadata_path=metadata_path,
        output_path=output_path
    )
    
    print(f"지도 생성 완료! 브라우저에서 {output_path} 파일을 열어 확인하세요.")


