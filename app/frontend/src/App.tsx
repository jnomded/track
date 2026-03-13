import { useState, useCallback, useRef } from 'react'
import Map, {
  Source,
  Layer,
  Marker,
  Popup,
  NavigationControl,
} from 'react-map-gl'
import type { MapRef } from 'react-map-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import './App.css'
import type { Detection, GeocodingFeature, ScanResult } from './types'

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN as string
const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

function circleGeoJSON(lat: number, lng: number, radiusKm: number, steps = 64) {
  const R = 6371
  const coords: [number, number][] = []
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * 2 * Math.PI
    const dlat = (radiusKm / R) * (180 / Math.PI) * Math.sin(angle)
    const dlng =
      ((radiusKm / R) * (180 / Math.PI) * Math.cos(angle)) /
      Math.cos((lat * Math.PI) / 180)
    coords.push([lng + dlng, lat + dlat])
  }
  return {
    type: 'Feature' as const,
    geometry: { type: 'Polygon' as const, coordinates: [coords] },
    properties: {},
  }
}

function confidenceColor(conf: number): string {
  return conf >= 0.85 ? '#00c48c' : '#f5a623'
}

export default function App() {
  const mapRef = useRef<MapRef>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [viewState, setViewState] = useState({
    longitude: -98.5795,
    latitude: 39.8283,
    zoom: 4,
  })
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState<GeocodingFeature[]>([])
  const [center, setCenter] = useState<{ lat: number; lng: number; name: string } | null>(null)
  const [radiusKm, setRadiusKm] = useState(5)
  const [threshold, setThreshold] = useState(65)
  const [scanning, setScanning] = useState(false)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [selectedMarker, setSelectedMarker] = useState<Detection | null>(null)

  const fetchSuggestions = useCallback((query: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (!query.trim()) {
      setSuggestions([])
      return
    }
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(
          `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json` +
            `?country=US&types=place,address,poi&limit=5&access_token=${MAPBOX_TOKEN}`
        )
        const data = await res.json()
        setSuggestions(data.features ?? [])
      } catch {
        setSuggestions([])
      }
    }, 300)
  }, [])

  const selectSuggestion = (feature: GeocodingFeature) => {
    const [lng, lat] = feature.center
    const shortName = feature.place_name.split(',')[0]
    setCenter({ lat, lng, name: shortName })
    setSearchQuery(shortName)
    setSuggestions([])
    setResult(null)
    setSelectedMarker(null)
    mapRef.current?.flyTo({ center: [lng, lat], zoom: 12, duration: 1500 })
  }

  const handleScan = async () => {
    if (!center || scanning) return
    setScanning(true)
    setResult(null)
    setSelectedMarker(null)
    try {
      const res = await fetch(`${API_URL}/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: center.lat,
          lng: center.lng,
          radius_km: radiusKm,
          threshold: threshold / 100,
        }),
      })
      if (!res.ok) throw new Error(`API error ${res.status}`)
      const data: ScanResult = await res.json()
      setResult(data)
    } catch (err) {
      console.error('Scan failed:', err)
    } finally {
      setScanning(false)
    }
  }

  const scanCircle = center ? circleGeoJSON(center.lat, center.lng, radiusKm) : null

  const scanBtnLabel = scanning
    ? 'Scanning...'
    : center
    ? `Scan ${radiusKm} km around ${center.name}`
    : 'Search for a location first'

  return (
    <div className="app">
      {/* ── Sidebar ─────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>TrackFinder</h1>
          <p>AI-powered running track detection</p>
        </div>

        <div className="sidebar-body">
          {/* Search */}
          <div className="search-container">
            <input
              className="search-input"
              placeholder="Search US location..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                fetchSuggestions(e.target.value)
              }}
              onKeyDown={(e) => e.key === 'Escape' && setSuggestions([])}
            />
            {suggestions.length > 0 && (
              <div className="suggestions">
                {suggestions.map((s, i) => (
                  <div key={i} className="suggestion-item" onClick={() => selectSuggestion(s)}>
                    {s.place_name}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Radius */}
          <div className="control-group">
            <div className="control-label">
              <span>Search radius</span>
              <span>{radiusKm} km</span>
            </div>
            <input
              className="slider"
              type="range"
              min={1}
              max={15}
              step={1}
              value={radiusKm}
              onChange={(e) => setRadiusKm(Number(e.target.value))}
            />
          </div>

          {/* Threshold */}
          <div className="control-group">
            <div className="control-label">
              <span>Confidence threshold</span>
              <span>{threshold}%</span>
            </div>
            <input
              className="slider"
              type="range"
              min={50}
              max={95}
              step={5}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>

          {/* Scan */}
          <button className="scan-btn" onClick={handleScan} disabled={!center || scanning}>
            {scanBtnLabel}
          </button>

          {/* Results */}
          {result && (
            <>
              <div className="stats-bar">
                <strong>{result.tiles_scanned}</strong> tiles scanned ·{' '}
                <strong>{result.detections.length}</strong> track
                {result.detections.length !== 1 ? 's' : ''} found
              </div>

              {result.detections.length === 0 ? (
                <p className="no-results">
                  No tracks detected in this area.
                  <br />
                  Try lowering the confidence threshold.
                </p>
              ) : (
                <>
                  <div className="results-label">Detections</div>
                  <div className="results-list">
                    {result.detections.map((d, i) => (
                      <div
                        key={i}
                        className={`result-item${selectedMarker === d ? ' active' : ''}`}
                        onClick={() => {
                          setSelectedMarker(d)
                          mapRef.current?.flyTo({
                            center: [d.lng, d.lat],
                            zoom: 15,
                            duration: 800,
                          })
                        }}
                      >
                        <div className="result-coords">
                          {d.lat.toFixed(4)}°, {d.lng.toFixed(4)}°
                        </div>
                        <div
                          className={`result-confidence${d.confidence < 0.85 ? ' medium' : ''}`}
                        >
                          {Math.round(d.confidence * 100)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </aside>

      {/* ── Map ─────────────────────────────────────── */}
      <div className="map-container">
        <Map
          ref={mapRef}
          {...viewState}
          onMove={(e) => setViewState(e.viewState)}
          mapStyle="mapbox://styles/mapbox/satellite-v9"
          mapboxAccessToken={MAPBOX_TOKEN}
          style={{ width: '100%', height: '100%' }}
          onClick={() => {
            setSuggestions([])
            setSelectedMarker(null)
          }}
        >
          <NavigationControl position="top-right" />

          {/* Scan radius circle */}
          {scanCircle && (
            <Source id="scan-radius" type="geojson" data={scanCircle}>
              <Layer
                id="scan-radius-fill"
                type="fill"
                paint={{
                  'fill-color': '#00c48c',
                  'fill-opacity': scanning ? 0.12 : 0.06,
                }}
              />
              <Layer
                id="scan-radius-outline"
                type="line"
                paint={{
                  'line-color': '#00c48c',
                  'line-width': 1.5,
                  'line-opacity': 0.5,
                }}
              />
            </Source>
          )}

          {/* Detection markers */}
          {result?.detections.map((d, i) => (
            <Marker key={i} longitude={d.lng} latitude={d.lat} anchor="center">
              <div
                style={{
                  width: 18,
                  height: 18,
                  borderRadius: '50%',
                  background: confidenceColor(d.confidence),
                  border: '2.5px solid rgba(255,255,255,0.9)',
                  cursor: 'pointer',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.6)',
                  transform: selectedMarker === d ? 'scale(1.3)' : 'scale(1)',
                  transition: 'transform 0.15s',
                }}
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedMarker(d)
                }}
              />
            </Marker>
          ))}

          {/* Selected marker popup */}
          {selectedMarker && (
            <Popup
              longitude={selectedMarker.lng}
              latitude={selectedMarker.lat}
              anchor="bottom"
              offset={14}
              onClose={() => setSelectedMarker(null)}
              closeButton
            >
              <div className="popup-content">
                <div
                  className={`popup-confidence${selectedMarker.confidence < 0.85 ? ' medium' : ''}`}
                >
                  {Math.round(selectedMarker.confidence * 100)}% confident
                </div>
                <div className="popup-subtitle">Running track detected</div>
                <a
                  className="popup-link"
                  href={`https://www.google.com/maps/@${selectedMarker.lat},${selectedMarker.lng},18z`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Open in Google Maps ↗
                </a>
              </div>
            </Popup>
          )}
        </Map>

        {scanning && <div className="scanning-badge">Scanning satellite imagery...</div>}
      </div>
    </div>
  )
}
