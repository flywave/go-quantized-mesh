package terrain

type TileJson struct {
	Tilejson    string      `json:"tilejson"`
	Name        *string     `json:"name,omitempty"`
	Version     string      `json:"version,omitempty"`
	Format      string      `json:"format,omitempty"`
	Attribution interface{} `json:"attribution,omitempty"`
	Scheme      string      `json:"scheme"`
	Tiles       string      `json:"tiles"`
	Minzoom     *int        `json:"minzoom,omitempty"`
	Maxzoom     *int        `json:"maxzoom,omitempty"`
	Bounds      []float64   `json:"bounds,omitempty"`
	Projection  string      `json:"projection"`
	Available   []int       `json:"available,omitempty"`
	Extensions  []string    `json:"extensions,omitempty"`
}

func NewTileJson(name string, minzoom int, maxzoom int, available []int, baseUrls string, flag TerrainExtensionFlag) *TileJson {
	var ext []string
	if (flag & Ext_Light) > 0 {
		ext = append(ext, "octvertexnormals")
		ext = append(ext, "vertexnormals")
	}
	if (flag & Ext_WaterMask) > 0 {
		ext = append(ext, "watermask")
	}
	if (flag & Ext_Metadata) > 0 {
		ext = append(ext, "metadata")
	}
	if len(available) == 0 {
		for i := minzoom; i < maxzoom; i++ {
			available = append(available, i)
		}
	}

	return &TileJson{Tilejson: "2.1.0", Name: &name, Version: "0.0.1", Format: "quantized-mesh-1.0", Scheme: "tms", Tiles: baseUrls, Minzoom: &minzoom, Maxzoom: &maxzoom, Bounds: []float64{-180, -90, 180, 90}, Projection: "EPSG:4326", Available: available, Extensions: ext}
}
