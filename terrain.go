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
	BITS12 = iota
	BITS16
)

const (
	QUANTIZED_COORDINATE_SIZE             = 32767
	QUANTIZED_BIT12_COORDINATE_SIZE       = 4095
	QUANTIZED_MESH_HEADER_SIZE            = 88
	QUANTIZED_MESH_LIGHT_EXTENSION_ID     = 1
	QUANTIZED_MESH_WATERMASK_EXTENSION_ID = 2
	QUANTIZED_MESH_METADATA_EXTENSION_ID  = 4
	QUANTIZED_MESH_WATERMASK_TILEPXS      = 65536
)

type TerrainExtensionFlag uint32

const (
	Ext_None            TerrainExtensionFlag = 0
	Ext_Light           TerrainExtensionFlag = 1
	Ext_WaterMask       TerrainExtensionFlag = 2
	Ext_Light_WaterMask TerrainExtensionFlag = Ext_Light | Ext_WaterMask
	Ext_Metadata        TerrainExtensionFlag = 4
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
	return BaseMime + strings.Join(ext, "-")
}

var (
	byteOrder = binary.LittleEndian
)

func unscaleCoordinate(v int, quantization int) float64 {
	switch quantization {
	case BITS12:
		return float64(v) / QUANTIZED_BIT12_COORDINATE_SIZE
	default: // BITS16
		return float64(v) / QUANTIZED_COORDINATE_SIZE
	}
}

func quantizeCoordinate(v float64, min float64, max float64, quantization int) int {
	delta := math.Abs(max - min)
	if delta == 0 {
		delta = 1
	}
	scaled := (v - min) / delta

	switch quantization {
	case BITS12:
		return int(scaled * QUANTIZED_BIT12_COORDINATE_SIZE) // 12 位最大值为 4095
	default: // 默认 16 位
		return int(scaled * QUANTIZED_COORDINATE_SIZE)
	}
}

func dequantizeCoordinate(v int, min float64, max float64, quantization int) float64 {
	delta := max - min
	switch quantization {
	case BITS12:
		return min + (float64(v)/QUANTIZED_BIT12_COORDINATE_SIZE)*delta
	default:
		return min + unscaleCoordinate(v, quantization)*delta
	}
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

// 修改 VertexData 结构
type VertexData struct {
	VertexCount uint32
	Data        []uint16 // 统一存储顶点数据
}

// 重写 Read 方法
func (vd *VertexData) Read(reader io.Reader, quantization int, minHeight, maxHeight float32) (int, error) {
	// 读取顶点数量
	if err := binary.Read(reader, binary.LittleEndian, &vd.VertexCount); err != nil {
		return 0, err
	}
	vertexCount := int(vd.VertexCount)

	// 根据量化类型确定数据大小
	elements := 3 // BITS16 模式
	if quantization == BITS12 {
		elements = 3 // TS 兼容布局: [xy, zh, uv]
	}

	// 读取顶点数据
	dataSize := vertexCount * elements
	vd.Data = make([]uint16, dataSize)
	if err := binary.Read(reader, binary.LittleEndian, vd.Data); err != nil {
		return 0, err
	}

	// 执行差分解码
	u, v, h := 0, 0, 0
	for i := 0; i < vertexCount; i++ {
		idx := i * elements

		// 解码并累加
		u += decodeZigZag(vd.Data[idx])
		vd.Data[idx] = uint16(u)

		v += decodeZigZag(vd.Data[idx+1])
		vd.Data[idx+1] = uint16(v)

		h += decodeZigZag(vd.Data[idx+2])
		vd.Data[idx+2] = uint16(h)
	}

	return 4 + dataSize*2, nil
}

// 重写 Write 方法
func (v *VertexData) Write(writer io.Writer, quantization int, minHeight, maxHeight float32) (int, error) {
	// 克隆数据以避免修改原始
	data := make([]uint16, len(v.Data))
	copy(data, v.Data)

	// 执行差分编码
	prevU, prevV, prevH := 0, 0, 0
	elements := 3
	for i := 0; i < int(v.VertexCount); i++ {
		idx := i * elements

		// 计算差分值
		u := int(data[idx]) - prevU
		v := int(data[idx+1]) - prevV
		h := int(data[idx+2]) - prevH

		// 存储差分值
		data[idx] = encodeZigZag(u)
		data[idx+1] = encodeZigZag(v)
		data[idx+2] = encodeZigZag(h)

		// 更新前值
		prevU = int(data[idx])
		prevV = int(data[idx+1])
		prevH = int(data[idx+2])
	}

	// 写入数据
	if err := binary.Write(writer, binary.LittleEndian, v.VertexCount); err != nil {
		return 0, err
	}
	if err := binary.Write(writer, binary.LittleEndian, data); err != nil {
		return 0, err
	}

	return 4 + len(data)*2, nil
}

func encodeZigZag(i int) uint16 {
	return uint16((i >> 15) ^ (i << 1))
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
	quantization int // 新增量化类型字段 0-BITS12,1-BITS16
}

type MeshData struct {
	BBox     [2][3]float64
	Vertices [][3]float64
	Normals  [][3]float64
	Faces    [][3]int
}

func NewMeshData() *MeshData {
	return &MeshData{
		BBox: [2][3]float64{vec3d.MaxVal, vec3d.MinVal},
	}
}

func (m *MeshData) AppendMesh(index int, mesh *tin.Mesh) {
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
	elements := 3

	for i := range vecs {
		idx := i * elements

		// 从数据缓冲区获取值
		u := int(t.Data.Data[idx])
		v := int(t.Data.Data[idx+1])
		h := int(t.Data.Data[idx+2])

		// 反量化坐标
		x := dequantizeCoordinate(u, bbox[0][0], bbox[1][0], t.quantization)
		y := dequantizeCoordinate(v, bbox[0][1], bbox[1][1], t.quantization)
		z := dequantizeCoordinate(h, bbox[0][2], bbox[1][2], t.quantization)

		vecs[i] = [3]float64{x, y, z}
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
	// 判断量化类型
	dimensions := vec3d.Sub((*vec3d.T)(&mesh.BBox[1]), (*vec3d.T)(&mesh.BBox[0]))
	maxDim := math.Max(math.Max(dimensions[0], dimensions[1]), dimensions[2])

	if maxDim < 4095 { // 2^12 - 1
		t.quantization = BITS12
	} else {
		t.quantization = BITS16
	}

	// 设置头部信息
	t.setHeader(mesh, rescaled)

	// 创建顶点数据缓冲区 (每个顶点3个uint16: u, v, height)
	elements := 3
	vertexCount := len(mesh.Vertices)
	data := make([]uint16, vertexCount*elements)

	// 处理每个顶点
	for i, vert := range mesh.Vertices {
		idx := i * elements

		// 量化坐标
		u := quantizeCoordinate(vert[0], mesh.BBox[0][0], mesh.BBox[1][0], t.quantization)
		v := quantizeCoordinate(vert[1], mesh.BBox[0][1], mesh.BBox[1][1], t.quantization)
		h := quantizeCoordinate(vert[2], mesh.BBox[0][2], mesh.BBox[1][2], t.quantization)

		// 存储原始量化值 (稍后执行差分编码)
		data[idx] = uint16(u)
		data[idx+1] = uint16(v)
		data[idx+2] = uint16(h)
	}

	// 设置顶点数据
	t.Data = VertexData{
		VertexCount: uint32(vertexCount),
		Data:        data,
	}

	// 处理索引和边界顶点
	var indices []int
	setflgs := make(map[int]int)
	index := 0

	// 边界顶点集合
	var northlings []uint32
	var eastlings []uint32
	var southlings []uint32
	var westlings []uint32

	maxCoord := QUANTIZED_COORDINATE_SIZE
	if t.quantization == BITS12 {
		maxCoord = QUANTIZED_BIT12_COORDINATE_SIZE
	}

	// 处理每个面
	for f := range mesh.Faces {
		for tf := range mesh.Faces[f] {
			i := mesh.Faces[f][tf]

			// 如果顶点已处理过，重用索引
			if k, ok := setflgs[i]; ok {
				indices = append(indices, k)
				continue
			}

			indices = append(indices, index)
			setflgs[i] = index

			// 检查边界顶点
			u := int(data[i*elements])
			v := int(data[i*elements+1])

			switch u {
			case 0:
				westlings = append(westlings, uint32(index))
			case maxCoord:
				eastlings = append(eastlings, uint32(index))
			}

			switch v {
			case 0:
				northlings = append(northlings, uint32(index))
			case maxCoord:
				southlings = append(southlings, uint32(index))
			}

			index++
		}
	}

	// 设置索引数据
	if vertexCount > 65535 {
		inds := make([]uint32, len(indices))
		for k, v := range indices {
			inds[k] = uint32(v)
		}
		t.Index = &Indices32{
			IndicesData: inds,
			northlings:  northlings,
			eastlings:   eastlings,
			southlings:  southlings,
			westlings:   westlings,
		}
	} else {
		inds := make([]uint16, len(indices))
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

		t.Index = &Indices16{
			IndicesData: inds,
			northlings:  nl,
			eastlings:   el,
			southlings:  sl,
			westlings:   wl,
		}
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
	if offset, err = t.Data.Read(reader, t.quantization, t.Header.MinimumHeight, t.Header.MaximumHeight); err != nil {
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

		switch lh.ExtensionLength {
		case 1:
			var mask uint8

			err := binary.Read(reader, byteOrder, &mask)
			if err != nil {
				return err
			}

			t.WaterMasks = &WaterMaskLand{Mask: mask}
		case QUANTIZED_MESH_WATERMASK_TILEPXS:
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
	return nil
}

func (t *QuantizedMeshTile) Write(writer io.Writer) error {
	var err error
	var offset int
	var alignment int
	if err = binary.Write(writer, byteOrder, t.Header); err != nil {
		return err
	}
	if offset, err = t.Data.Write(writer, t.quantization, t.Header.MinimumHeight, t.Header.MaximumHeight); err != nil {
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

	return nil
}
