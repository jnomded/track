export interface Detection {
  lat: number;
  lng: number;
  confidence: number;
}

export interface GeocodingFeature {
  place_name: string;
  center: [number, number]; // [lng, lat]
}

export interface ScanResult {
  detections: Detection[];
  tiles_scanned: number;
}
