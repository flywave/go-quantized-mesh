package terrain

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strings"
	"unsafe"

	"github.com/flywave/go-proj"
	tin "github.com/flywave/go-tin"
	vec3d "github.com/flywave/go3d/float64/vec3"
)

const (
	QUANTIZED_COORDINATE_SIZE             = 32767
	QUANTIZED_MESH_HEADER_SIZE            = 88
	QUANTIZED_MESH_LIGHT_EXTENSION_ID     = 1
	QUANTIZED_MESH_WATERMASK_EXTENSION_ID = 2
	QUANTIZED_MESH_METADATA_EXTENSION_ID  = 4
	QUANTIZED_MESH_FACEGROUP_EXTENSION_ID = 8
	QUANTIZED_MESH_WATERMASK_TILEPXS      = 65536
)

type TerrainExtensionFlag uint32

const (
	Ext_None            TerrainExtensionFlag = 0
	Ext_Light           TerrainExtensionFlag = 1
	Ext_WaterMask       TerrainExtensionFlag = 2
	Ext_Light_WaterMask TerrainExtensionFlag = Ext_Light | Ext_WaterMask
	Ext_Metadata        TerrainExtensionFlag = 4
	Ext_FaceGroup       TerrainExtensionFlag = 8
)

const llh_ecef_radiusX = 6378137.0
const llh_ecef_radiusY = 6378137.0
const llh_ecef_radiusZ = 6356752.3142451793

const llh_ecef_rX = 1.0 / llh_ecef_radiusX
const llh_ecef_rY = 1.0 / llh_ecef_radiusY
const llh_ecef_rZ = 1.0 / llh_ecef_radiusZ

const BaseMime = "application/vnd.quantized-mesh;extensions="

func GetTerrainMime(flag TerrainExtensionFlag) string {
	var ext []string
	if (flag & Ext_Light) > 0 {
		ext = append(ext, "octvertexnormals")
	}
	if (flag & Ext_WaterMask) > 0 {
		ext = append(ext, "watermask")
	}
	if (flag & Ext_Metadata) > 0 {
		ext = append(ext, "metadata")
	}
	if (flag & Ext_FaceGroup) > 0 {
		ext = append(ext, "facegroup")
	}
	return BaseMime + strings.Join(ext, "-")
}

var (
	byteOrder = binary.LittleEndian
)

func scaleCoordinate(v float64) int {
	return int(v * QUANTIZED_COORDINATE_SIZE)
}

func unscaleCoordinate(v int) float64 {
	return float64(v) / QUANTIZED_COORDINATE_SIZE
}

func quantizeCoordinate(v float64, min float64, max float64) int {
	delta := math.Abs(max - min)
	if delta == 0 {
		delta = 1
	}
	return scaleCoordinate(float64(v-min) / delta)
}

func dequantizeCoordinate(v int, min float64, max float64) float64 {
	delta := max - min
	return min + unscaleCoordinate(v)*delta
}

type QuantizedMeshHeader struct {
	CenterX float64
	CenterY float64
	CenterZ float64

	MinimumHeight float32
	MaximumHeight float32

	BoundingSphereCenterX float64
	BoundingSphereCenterY float64
	BoundingSphereCenterZ float64
	BoundingSphereRadius  float64

	HorizonOcclusionPointX float64
	HorizonOcclusionPointY float64
	HorizonOcclusionPointZ float64
}

type VertexData struct {
	VertexCount uint32
	U           []uint16
	V           []uint16
	H           []uint16
}

func (v *VertexData) Read(reader io.Reader) (int, error) {
	offset := 0
	buf := make([]byte, 4)
	offset += 4
	if _, err := reader.Read(buf); err != nil {
		return 0, err
	}
	v.VertexCount = byteOrder.Uint32(buf)

	buf = make([]byte, v.VertexCount*2)
	if _, err := reader.Read(buf); err != nil {
		return 0, err
	}
	v.U = v.decodeArray(buf, int(v.VertexCount))

	buf = make([]byte, v.VertexCount*2)
	if _, err := reader.Read(buf); err != nil {
		return 0, err
	}
	v.V = v.decodeArray(buf, int(v.VertexCount))

	buf = make([]byte, v.VertexCount*2)
	if _, err := reader.Read(buf); err != nil {
		return 0, err
	}
	v.H = v.decodeArray(buf, int(v.VertexCount))
	offset += int(2*v.VertexCount + 2*v.VertexCount + 2*v.VertexCount)
	return offset, nil
}

func (v *VertexData) Write(writer io.Writer) (int, error) {
	buf := make([]byte, 4)
	offset := 0
	byteOrder.PutUint32(buf, v.VertexCount)
	offset += 4
	if _, err := writer.Write(buf); err != nil {
		return 0, err
	}
	buf = v.encodeArray(v.U, int(v.VertexCount))
	if _, err := writer.Write(buf); err != nil {
		return 0, err
	}
	buf = v.encodeArray(v.V, int(v.VertexCount))
	if _, err := writer.Write(buf); err != nil {
		return 0, err
	}
	buf = v.encodeArray(v.H, int(v.VertexCount))
	if _, err := writer.Write(buf); err != nil {
		return 0, err
	}
	offset += int(2*v.VertexCount + 2*v.VertexCount + 2*v.VertexCount)
	return offset, nil
}

func (v *VertexData) encodeArray(values []uint16, vertexCount int) []byte {
	buf := make([]byte, vertexCount*2)
	for i := 0; i < vertexCount; i++ {
		byteOrder.PutUint16(buf[i*2:i*2+2], values[i])
	}
	return buf
}

func encodeZigZag(i int) uint16 {
	return uint16((i >> 15) ^ (i << 1))
}

func (v *VertexData) decodeArray(buffer []byte, vertexCount int) []uint16 {
	values := make([]uint16, vertexCount)
	for i := 0; i < vertexCount; i++ {
		values[i] = byteOrder.Uint16(buffer[i*2 : i*2+2])
	}
	return values
}

func decodeZigZag(encoded uint16) int {
	unsignedEncoded := int(encoded)
	return unsignedEncoded>>1 ^ -(unsignedEncoded & 1)
}

func calcPadding(offset, paddingUnit int) int {
	padding := offset % paddingUnit
	if padding != 0 {
		padding = paddingUnit - padding
	}
	return padding
}

type Indices interface {
	GetIndexCount() int
	GetIndex(i int) int
	CalcSize() int64
	Read(reader io.Reader) error
	Write(writer io.Writer) error
}

type Indices16 struct {
	IndicesData []uint16
	northlings  []uint16
	eastlings   []uint16
	southlings  []uint16
	westlings   []uint16
}

func (ind *Indices16) GetIndexCount() int {
	return len(ind.IndicesData)
}

func (ind *Indices16) GetIndex(i int) int {
	return int(ind.IndicesData[i])
}

func (ind *Indices16) CalcSize() int64 {
	return int64(4 + ind.GetIndexCount()*2)
}

func (ind *Indices16) decodeIndices(indices []uint16) []uint16 {
	highest := uint16(0)
	for i := 0; i < len(indices); i++ {
		code := indices[i]
		indices[i] = uint16(highest - code)
		if code == 0 {
			highest++
		}
	}
	return indices
}

func (ind *Indices16) encodeIndices(indices []uint16) []uint16 {
	watermark := uint16(0)
	for i := 0; i < len(indices); i++ {
		code := indices[i]
		delta := watermark - code
		indices[i] = delta
		if code == watermark {
			watermark++
		}
	}
	return indices
}

func (ind *Indices16) Read(reader io.Reader) error {
	buf := make([]byte, 4)
	var triangleCount uint32
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	triangleCount = byteOrder.Uint32(buf)

	ind.IndicesData = make([]uint16, triangleCount*3)
	err := binary.Read(reader, byteOrder, ind.IndicesData)
	if err != nil {
		return err
	}
	ind.IndicesData = ind.decodeIndices(ind.IndicesData)

	ind.westlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	ind.southlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	ind.eastlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	ind.northlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	return nil
}

func (ind *Indices16) ReadIndices(reader io.Reader) ([]uint16, error) {
	count := uint32(0)
	err := binary.Read(reader, byteOrder, &count)
	if err != nil {
		return nil, err
	}
	indices := make([]uint16, count)

	err = binary.Read(reader, byteOrder, indices)
	if err != nil {
		return nil, err
	}
	return indices, nil
}

func (ind *Indices16) WriteIndices(writer io.Writer, indices []uint16) error {
	err := binary.Write(writer, byteOrder, uint32(len(indices)))
	if err != nil {
		return err
	}

	err = binary.Write(writer, byteOrder, indices)
	if err != nil {
		return err
	}
	return nil
}

func (ind *Indices16) Write(writer io.Writer) error {
	data := ind.encodeIndices(ind.IndicesData)
	err := binary.Write(writer, byteOrder, uint32(ind.GetIndexCount()/3))
	if err != nil {
		return err
	}
	err = binary.Write(writer, byteOrder, data)
	if err != nil {
		return err
	}

	err = ind.WriteIndices(writer, ind.westlings)
	if err != nil {
		return err
	}
	err = ind.WriteIndices(writer, ind.southlings)
	if err != nil {
		return err
	}
	err = ind.WriteIndices(writer, ind.eastlings)
	if err != nil {
		return err
	}
	err = ind.WriteIndices(writer, ind.northlings)
	if err != nil {
		return err
	}
	return nil
}

type Indices32 struct {
	IndicesData []uint32
	northlings  []uint32
	eastlings   []uint32
	southlings  []uint32
	westlings   []uint32
}

func (ind *Indices32) GetIndexCount() int {
	return len(ind.IndicesData)
}

func (ind *Indices32) GetIndex(i int) int {
	return int(ind.IndicesData[i])
}

func (ind *Indices32) CalcSize() int64 {
	return int64(4 + ind.GetIndexCount()*4)
}

func (ind *Indices32) decodeIndices(indices []uint32) []uint32 {
	highest := uint32(0)
	for i := 0; i < len(indices); i++ {
		code := indices[i]
		indices[i] = uint32(highest - code)
		if code == 0 {
			highest++
		}
	}
	return indices
}

func (ind *Indices32) encodeIndices(indices []uint32) []uint32 {
	highest := uint32(0)
	for i := 0; i < len(indices); i++ {
		code := indices[i]
		delta := highest - code
		indices[i] = delta
		if code == highest {
			highest++
		}
	}
	return indices
}

func (ind *Indices32) Read(reader io.Reader) error {
	buf := make([]byte, 4)
	var triangleCount uint32
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	triangleCount = byteOrder.Uint32(buf)
	ind.IndicesData = make([]uint32, triangleCount*3)
	err := binary.Read(reader, byteOrder, ind.IndicesData)
	if err != nil {
		return err
	}
	ind.IndicesData = ind.decodeIndices(ind.IndicesData)

	ind.westlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	ind.southlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	ind.eastlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	ind.northlings, err = ind.ReadIndices(reader)
	if err != nil {
		return err
	}
	return nil
}

func (ind *Indices32) ReadIndices(reader io.Reader) ([]uint32, error) {
	count := uint32(0)
	err := binary.Read(reader, byteOrder, &count)
	if err != nil {
		return nil, err
	}
	indices := make([]uint32, count)

	err = binary.Read(reader, byteOrder, indices)
	if err != nil {
		return nil, err
	}
	return indices, nil
}

func (ind *Indices32) WriteIndices(writer io.Writer, indices []uint32) error {
	err := binary.Write(writer, byteOrder, uint32(len(indices)))
	if err != nil {
		return err
	}

	err = binary.Write(writer, byteOrder, indices)
	if err != nil {
		return err
	}
	return nil
}

func (ind *Indices32) Write(writer io.Writer) error {
	data := ind.encodeIndices(ind.IndicesData)
	err := binary.Write(writer, byteOrder, uint32(ind.GetIndexCount()/3))
	if err != nil {
		return err
	}
	err = binary.Write(writer, byteOrder, data)
	if err != nil {
		return err
	}

	err = ind.WriteIndices(writer, ind.westlings)
	if err != nil {
		return err
	}
	err = ind.WriteIndices(writer, ind.southlings)
	if err != nil {
		return err
	}
	err = ind.WriteIndices(writer, ind.eastlings)
	if err != nil {
		return err
	}
	err = ind.WriteIndices(writer, ind.northlings)
	if err != nil {
		return err
	}
	return nil
}

type QuantizedMeshTile struct {
	Header       QuantizedMeshHeader
	Data         VertexData
	Index        interface{}
	Edge         interface{}
	LightNormals *OctEncodedVertexNormals
	WaterMasks   interface{}
	Metadata     *Metadata
	FaceGroop    []*FaceGroop
}

type MeshData struct {
	BBox      [2][3]float64
	Vertices  [][3]float64
	Normals   [][3]float64
	Faces     [][3]int
	FaceGroop []*FaceGroop
}

func NewMeshData() *MeshData {
	return &MeshData{
		BBox: [2][3]float64{vec3d.MinVal, vec3d.MinVal},
	}
}

func (m *MeshData) AppendMesh(mesh *tin.Mesh) {
	g := &FaceGroop{Start: uint32(len(m.Faces))}
	m.BBox[0] = vec3d.Min((*vec3d.T)(&m.BBox[0]), (*vec3d.T)(&mesh.BBox[0]))
	m.BBox[1] = vec3d.Max((*vec3d.T)(&m.BBox[1]), (*vec3d.T)(&mesh.BBox[1]))

	count := len(m.Vertices)
	for _, f := range mesh.Faces {
		m.Faces = append(m.Faces, [3]int{count + int(f[0]), count + int(f[1]), count + int(f[2])})
	}

	vts := *(*[][3]float64)(unsafe.Pointer(&mesh.Vertices))
	m.Vertices = append(m.Vertices, vts...)

	nls := *(*[][3]float64)(unsafe.Pointer(&mesh.Normals))
	m.Normals = append(m.Normals, nls...)

	g.End = uint32(len(m.Faces))
	m.FaceGroop = append(m.FaceGroop, g)
}

func distance(max, min []float64) float64 {
	x := max[0] - min[0]
	y := max[1] - min[1]
	return math.Sqrt(x*x + y*y)
}

func (t *QuantizedMeshTile) setHeader(mesh *MeshData, rescaled bool) {
	bbox := mesh.BBox
	t.Header.MinimumHeight = float32(bbox[0][2])
	t.Header.MaximumHeight = float32(bbox[1][2])

	bbox[0][0], bbox[0][1], bbox[0][2], _ = proj.Lonlat2Ecef(bbox[0][0], bbox[0][1], bbox[0][2])
	bbox[1][0], bbox[1][1], bbox[1][2], _ = proj.Lonlat2Ecef(bbox[1][0], bbox[1][1], bbox[1][2])
	var c [3]float64
	c[0] = (bbox[1][0] + bbox[0][0]) / 2
	c[1] = (bbox[1][1] + bbox[0][1]) / 2
	c[2] = (bbox[1][2] + bbox[0][2]) / 2

	t.Header.CenterX = c[0]
	t.Header.CenterY = c[1]
	t.Header.CenterZ = c[2]

	t.Header.BoundingSphereCenterX = c[0]
	t.Header.BoundingSphereCenterY = c[1]
	t.Header.BoundingSphereCenterZ = c[2]
	t.Header.BoundingSphereRadius = distance(bbox[1][0:2], bbox[0][0:2]) / 2

	t.ocp_fromPoints(mesh, rescaled)
	t.Header.HorizonOcclusionPointX = c[0]
	t.Header.HorizonOcclusionPointY = c[1]
	t.Header.HorizonOcclusionPointZ = c[2]
}

func (t *QuantizedMeshTile) ocp_fromPoints(mesh *MeshData, rescaled bool) [3]float64 {
	scaledCenter := [3]float64{t.Header.BoundingSphereCenterX * llh_ecef_rX, t.Header.BoundingSphereCenterY * llh_ecef_rY, t.Header.BoundingSphereCenterZ * llh_ecef_rZ}
	min := mesh.BBox[0]
	max_magnitude := -math.MaxFloat64
	for _, v := range mesh.Vertices {
		if rescaled {
			v[0] += min[0]
			v[1] += min[1]
			v[2] += min[2]
		}
		x, y, z, _ := proj.Lonlat2Ecef(v[0], v[1], v[2])
		scaledPt := [3]float64{x * llh_ecef_rX, y * llh_ecef_rY, z * llh_ecef_rZ}
		magnitude := ocp_computeMagnitude(scaledPt, scaledCenter)
		if magnitude > max_magnitude {
			max_magnitude = magnitude
		}
	}
	sc := vec3d.T(scaledCenter)
	sc.Scale(max_magnitude)
	return sc
}

func ocp_computeMagnitude(position vec3d.T, sphereCenter vec3d.T) float64 {
	magnitudeSquared := position.LengthSqr()
	magnitude := math.Sqrt(magnitudeSquared)
	direction := position.Scale(1 / magnitude)

	// For the purpose of this computation, points below the ellipsoid
	// are considered to be on it instead.
	magnitudeSquared = math.Max(1.0, magnitudeSquared)
	magnitude = math.Max(1.0, magnitude)

	cosAlpha := vec3d.Dot(direction, &sphereCenter)
	sv := vec3d.Cross(direction, &sphereCenter)
	sinAlpha := sv.Length()
	cosBeta := 1.0 / magnitude
	sinBeta := math.Sqrt(magnitudeSquared-1.0) * cosBeta

	return 1.0 / (cosAlpha*cosBeta - sinAlpha*sinBeta)
}

func (t *QuantizedMeshTile) getBBoxFromHeader() [2][3]float64 {
	tl :=
		[2]float64{t.Header.BoundingSphereCenterX - t.Header.BoundingSphereRadius,
			t.Header.BoundingSphereCenterY - t.Header.BoundingSphereRadius}
	br := [2]float64{t.Header.BoundingSphereCenterX + t.Header.BoundingSphereRadius,
		t.Header.BoundingSphereCenterY + t.Header.BoundingSphereRadius}

	return [2][3]float64{
		{tl[0], tl[1], float64(t.Header.MinimumHeight)},
		{br[0], br[1], float64(t.Header.MaximumHeight)},
	}
}

func (t *QuantizedMeshTile) getVertices(bbox [2][3]float64) [][3]float64 {
	vecs := make([][3]float64, t.Data.VertexCount)
	u := 0
	v := 0
	height := 0
	for i := range vecs {
		u += decodeZigZag(t.Data.U[i])
		v += decodeZigZag(t.Data.V[i])
		height += decodeZigZag(t.Data.H[i])

		x := dequantizeCoordinate(u, bbox[0][0], bbox[1][0])
		y := dequantizeCoordinate(v, bbox[0][1], bbox[1][1])
		z := dequantizeCoordinate(height, bbox[0][2], bbox[1][2])

		vecs[i][0] = x
		vecs[i][1] = y
		vecs[i][2] = z
	}
	return vecs
}

func (t *QuantizedMeshTile) getFaces() ([][3]int, error) {
	if t.Index.(Indices).GetIndexCount()%3 != 0 {
		return nil, errors.New("mesh face error")
	}
	tri := t.Index.(Indices).GetIndexCount() / 3
	inds := make([][3]int, tri)
	for i := 0; i < tri; i++ {
		inds[i][0] = t.Index.(Indices).GetIndex(i * 3)
		inds[i][1] = t.Index.(Indices).GetIndex(i*3 + 1)
		inds[i][2] = t.Index.(Indices).GetIndex(i*3 + 2)
	}
	return inds, nil
}

func (t *QuantizedMeshTile) GetMesh() (*MeshData, error) {
	if t.Data.VertexCount < 3 {
		return nil, errors.New("mesh error")
	}
	bbox := t.getBBoxFromHeader()
	vertices := t.getVertices(bbox)
	if faces, err := t.getFaces(); err != nil {
		return nil, err
	} else {
		return &MeshData{BBox: bbox, Vertices: vertices, Faces: faces}, nil
	}
}

func (t *QuantizedMeshTile) SetMesh(mesh *MeshData, rescaled bool) {
	t.setHeader(mesh, rescaled)

	var us []uint16
	var vs []uint16
	var hs []uint16

	var northlings []uint32
	var eastlings []uint32
	var southlings []uint32
	var westlings []uint32

	u := 0
	v := 0
	h := 0
	prevu := 0
	prevv := 0
	prevh := 0

	indices := []int{}
	setflgs := make(map[int]int)
	index := 0

	for f := range mesh.Faces {
		for t := range mesh.Faces[f] {
			i := mesh.Faces[f][t]
			if k, ok := setflgs[i]; ok {
				indices = append(indices, k)
				continue
			}
			indices = append(indices, index)
			setflgs[i] = index

			if rescaled {
				u = scaleCoordinate(mesh.Vertices[i][0])
				v = scaleCoordinate(mesh.Vertices[i][1])
				h = scaleCoordinate(mesh.Vertices[i][2])
			} else {
				u = quantizeCoordinate(mesh.Vertices[i][0], mesh.BBox[0][0], mesh.BBox[1][0])
				v = quantizeCoordinate(mesh.Vertices[i][1], mesh.BBox[0][1], mesh.BBox[1][1])
				h = quantizeCoordinate(mesh.Vertices[i][2], mesh.BBox[0][2], mesh.BBox[1][2])
			}

			if u == 0 {
				westlings = append(westlings, uint32(index))
			} else if u == QUANTIZED_COORDINATE_SIZE {
				eastlings = append(eastlings, uint32(index))
			}

			if v == 0 {
				northlings = append(northlings, uint32(index))
			} else if v == QUANTIZED_COORDINATE_SIZE {
				southlings = append(southlings, uint32(index))
			}

			us = append(us, encodeZigZag(u-prevu))
			vs = append(vs, encodeZigZag(v-prevv))
			hs = append(hs, encodeZigZag(h-prevh))

			prevu = u
			prevv = v
			prevh = h
			index++
		}
	}

	t.Data.VertexCount = uint32(len(mesh.Vertices))
	t.Data.U = us
	t.Data.V = vs
	t.Data.H = hs

	if t.Data.VertexCount > 65535 {
		inds := make([]uint32, len(mesh.Faces)*3)
		for k, v := range indices {
			inds[k] = uint32(v)
		}
		t.Index = &Indices32{IndicesData: inds, northlings: northlings, eastlings: eastlings, southlings: southlings, westlings: westlings}
	} else {
		inds := make([]uint16, len(mesh.Faces)*3)
		for k, v := range indices {
			inds[k] = uint16(v)
		}

		nl := make([]uint16, len(northlings))
		for i := range northlings {
			nl[i] = uint16(northlings[i])
		}
		el := make([]uint16, len(eastlings))
		for i := range eastlings {
			el[i] = uint16(eastlings[i])
		}

		sl := make([]uint16, len(southlings))
		for i := range southlings {
			sl[i] = uint16(southlings[i])
		}
		wl := make([]uint16, len(westlings))
		for i := range westlings {
			wl[i] = uint16(westlings[i])
		}

		t.Index = &Indices16{IndicesData: inds, northlings: nl, eastlings: el, southlings: sl, westlings: wl}
	}
	if len(mesh.Normals) > 0 {
		nl := (*[]vec3d.T)(unsafe.Pointer(&mesh.Normals))
		t.LightNormals = &OctEncodedVertexNormals{Norm: *nl}
	}
}

func (t *QuantizedMeshTile) Read(reader io.ReadSeeker, flag TerrainExtensionFlag) error {
	var offset int
	err := binary.Read(reader, byteOrder, &t.Header)
	if err != nil {
		return err
	}
	if offset, err = t.Data.Read(reader); err != nil {
		return err
	}
	var alignment int
	if t.Data.VertexCount > 65535 {
		alignment = 4
	} else {
		alignment = 2
	}
	padding := calcPadding(QUANTIZED_MESH_HEADER_SIZE+offset, alignment)
	reader.Seek(int64(padding), io.SeekCurrent)
	if t.Data.VertexCount > 65535 {
		idx := new(Indices32)
		if err := idx.Read(reader); err != nil {
			return err
		}
		t.Index = idx
	} else {
		idx := new(Indices16)
		if err := idx.Read(reader); err != nil {
			return err
		}
		t.Index = idx
	}

	if (flag & Ext_Light) > 0 {
		lh := ExtensionHeader{}

		err := binary.Read(reader, byteOrder, &lh)
		if err != nil {
			return err
		}

		enN := make([]uint8, lh.ExtensionLength)

		err = binary.Read(reader, byteOrder, &enN)
		if err != nil {
			return err
		}

		t.LightNormals = &OctEncodedVertexNormals{Norm: make([]vec3d.T, int(lh.ExtensionLength/2))}
		for i := 0; i < int(lh.ExtensionLength/2); i++ {
			t.LightNormals.Norm[i] = octDecode(enN[i*2], enN[i*2+1])
		}
	}

	if (flag & Ext_WaterMask) > 0 {
		lh := ExtensionHeader{}

		err := binary.Read(reader, byteOrder, &lh)
		if err != nil {
			return err
		}

		if lh.ExtensionLength == 1 {
			var mask uint8

			err := binary.Read(reader, byteOrder, &mask)
			if err != nil {
				return err
			}

			t.WaterMasks = &WaterMaskLand{Mask: mask}
		} else if lh.ExtensionLength == QUANTIZED_MESH_WATERMASK_TILEPXS {
			masks := &WaterMask{}
			err := binary.Read(reader, byteOrder, &masks.Mask)
			if err != nil {
				return err
			}

			t.WaterMasks = masks
		}
	}

	if (flag & Ext_Metadata) > 0 {
		lh := ExtensionHeader{}

		err := binary.Read(reader, byteOrder, &lh)
		if err != nil {
			return err
		}

		var jsonLen uint32

		err = binary.Read(reader, byteOrder, &jsonLen)
		if err != nil {
			return err
		}

		js := make(json.RawMessage, jsonLen)

		err = binary.Read(reader, byteOrder, &js)
		if err != nil {
			return err
		}

		t.Metadata = &Metadata{Json: js}
	}

	if (flag & Ext_FaceGroup) > 0 {
		var count uint32
		err = binary.Read(reader, byteOrder, &count)
		if err != nil {
			return err
		}

		bt := make([]byte, count)
		err = binary.Read(reader, byteOrder, &bt)
		if err != nil {
			return err
		}
		fg := []*FaceGroop{}
		json.Unmarshal(bt, &fg)
		t.FaceGroop = fg
	}
	return nil
}

func (t *QuantizedMeshTile) Write(writer io.Writer) error {
	var err error
	var offset int
	var alignment int
	if err = binary.Write(writer, byteOrder, t.Header); err != nil {
		return err
	}
	if offset, err = t.Data.Write(writer); err != nil {
		return err
	}
	if t.Data.VertexCount > 65535 {
		alignment = 4
	} else {
		alignment = 2
	}
	padding := calcPadding(QUANTIZED_MESH_HEADER_SIZE+offset, alignment)
	if padding > 0 {
		buf := make([]byte, padding)
		for i := 0; i < padding; i++ {
			buf[i] = 0xCA
		}
		if _, err := writer.Write(buf); err != nil {
			return err
		}
	}
	switch ti := t.Index.(type) {
	case *Indices16:
		if err = ti.Write(writer); err != nil {
			return err
		}
	case *Indices32:
		if err = ti.Write(writer); err != nil {
			return err
		}
	default:
		return errors.New("index is not set")
	}

	if t.LightNormals != nil && len(t.LightNormals.Norm) == int(t.Data.VertexCount) {
		lhead := EXT_LIGHT_HEADER
		lhead.ExtensionLength = 2 * t.Data.VertexCount

		if err = binary.Write(writer, byteOrder, lhead); err != nil {
			return err
		}

		for _, n := range t.LightNormals.Norm {
			en := octEncode(n)

			if _, err := writer.Write(en[:]); err != nil {
				return err
			}
		}
	}

	if t.WaterMasks != nil {
		lhead := EXT_WATERMASK_HEADER
		switch wm := t.WaterMasks.(type) {
		case *WaterMaskLand:
			lhead.ExtensionLength = 1

			if err = binary.Write(writer, byteOrder, lhead); err != nil {
				return err
			}

			if _, err := writer.Write([]byte{byte(wm.Mask)}); err != nil {
				return err
			}

		case *WaterMask:
			lhead.ExtensionLength = QUANTIZED_MESH_WATERMASK_TILEPXS

			if err = binary.Write(writer, byteOrder, lhead); err != nil {
				return err
			}

			if err = binary.Write(writer, byteOrder, wm.Mask); err != nil {
				return err
			}
		}
	}

	if t.Metadata != nil {
		lhead := EXT_METADATA_HEADER

		if err = binary.Write(writer, byteOrder, lhead); err != nil {
			return err
		}

		jsonLen := uint32(len(t.Metadata.Json))

		if err = binary.Write(writer, byteOrder, jsonLen); err != nil {
			return err
		}

		if _, err := writer.Write(t.Metadata.Json); err != nil {
			return err
		}
	}

	if len(t.FaceGroop) > 0 {
		lhead := EXT_FACEGROUP_HEADER

		if err = binary.Write(writer, byteOrder, lhead); err != nil {
			return err
		}

		bt, _ := json.Marshal(t.FaceGroop)

		if err = binary.Write(writer, byteOrder, uint32(len(bt))); err != nil {
			return err
		}

		if _, err = writer.Write(bt); err != nil {
			return err
		}
	}
	return nil
}
