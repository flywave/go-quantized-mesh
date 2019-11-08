// Copyright (c) 2017-present FlyWave, Inc. All Rights Reserved.
// See License.txt for license information.

package terrain

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
)

const (
	QUANTIZED_COORDINATE_SIZE  = 32767
	QUANTIZED_MESH_HEADER_SIZE = 88
)

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
	delta := max - min
	if delta < 0 {
		return 0
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

func (h *QuantizedMeshHeader) Read(reader io.Reader) error {
	buf := make([]byte, 8)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.CenterX = math.Float64frombits(byteOrder.Uint64(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.CenterY = math.Float64frombits(byteOrder.Uint64(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.CenterZ = math.Float64frombits(byteOrder.Uint64(buf))
	buf = make([]byte, 4)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.MinimumHeight = math.Float32frombits(byteOrder.Uint32(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.MaximumHeight = math.Float32frombits(byteOrder.Uint32(buf))
	buf = make([]byte, 8)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.BoundingSphereCenterX = math.Float64frombits(byteOrder.Uint64(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.BoundingSphereCenterY = math.Float64frombits(byteOrder.Uint64(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.BoundingSphereCenterZ = math.Float64frombits(byteOrder.Uint64(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.BoundingSphereRadius = math.Float64frombits(byteOrder.Uint64(buf))

	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.HorizonOcclusionPointX = math.Float64frombits(byteOrder.Uint64(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.HorizonOcclusionPointY = math.Float64frombits(byteOrder.Uint64(buf))
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	h.HorizonOcclusionPointZ = math.Float64frombits(byteOrder.Uint64(buf))
	return nil
}

func (h *QuantizedMeshHeader) Write(writer io.Writer) error {
	buf := make([]byte, QUANTIZED_MESH_HEADER_SIZE)
	offset := 0
	bs := math.Float64bits(h.CenterX)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.CenterY)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.CenterZ)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs2 := math.Float32bits(h.MinimumHeight)
	byteOrder.PutUint32(buf[offset:offset+4], bs2)
	offset += 4
	bs2 = math.Float32bits(h.MaximumHeight)
	byteOrder.PutUint32(buf[offset:offset+4], bs2)
	offset += 4
	bs = math.Float64bits(h.BoundingSphereCenterX)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.BoundingSphereCenterY)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.BoundingSphereCenterZ)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.BoundingSphereRadius)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.HorizonOcclusionPointX)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.HorizonOcclusionPointY)
	byteOrder.PutUint64(buf[offset:offset+8], bs)
	offset += 8
	bs = math.Float64bits(h.HorizonOcclusionPointZ)
	byteOrder.PutUint64(buf[offset:], bs)
	offset += 8
	if _, err := writer.Write(buf); err != nil {
		return nil
	}
	return nil
}

type VertexData struct {
	VertexCount uint32
	U           []uint16
	V           []uint16
	H           []uint16
}

func (v *VertexData) Read(reader io.Reader) error {
	buf := make([]byte, 4)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	v.VertexCount = byteOrder.Uint32(buf)

	buf = make([]byte, v.VertexCount*2)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	v.U = v.decodeArray(buf, int(v.VertexCount))

	buf = make([]byte, v.VertexCount*2)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	v.V = v.decodeArray(buf, int(v.VertexCount))

	buf = make([]byte, v.VertexCount*2)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	v.H = v.decodeArray(buf, int(v.VertexCount))
	return nil
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
	value := 0
	for i := 0; i < vertexCount; i++ {
		t := values[i]
		byteOrder.PutUint16(buf[i:i+2], encodeZigZag(int(values[i])-value))
		value = int(t)
	}
	return buf
}

func encodeZigZag(i int) uint16 {
	return uint16((i >> 15) ^ (i << 1))
}

func (v *VertexData) decodeArray(buffer []byte, vertexCount int) []uint16 {
	values := make([]uint16, vertexCount)
	value := 0
	for i := 0; i < vertexCount; i++ {
		value += decodeZigZag(byteOrder.Uint16(buffer[i : i+2]))
		values[i] = uint16(value)
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
	Indices
	IndicesData []uint16
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
	indicesCount := triangleCount * 3
	buf = make([]byte, indicesCount*2)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	ind.IndicesData = make([]uint16, indicesCount)
	for i := range ind.IndicesData {
		ind.IndicesData[i] = uint16(byteOrder.Uint16(buf[i*2 : (i+1)*2]))
	}
	ind.IndicesData = ind.decodeIndices(ind.IndicesData)
	return nil
}

func (ind *Indices16) Write(writer io.Writer) error {
	data := ind.encodeIndices(ind.IndicesData)
	buf := make([]byte, ind.CalcSize())
	offset := 0
	byteOrder.PutUint32(buf[offset:offset+4], uint32(ind.GetIndexCount()/3))
	offset += 4
	for i := range data {
		byteOrder.PutUint16(buf[offset:offset+2], data[i])
		offset += 2
	}
	if _, err := writer.Write(buf); err != nil {
		return nil
	}
	return nil
}

type Indices32 struct {
	Indices
	IndicesData []uint32
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
	indicesCount := triangleCount * 3
	buf = make([]byte, indicesCount*4)
	if _, err := reader.Read(buf); err != nil {
		return err
	}
	ind.IndicesData = make([]uint32, indicesCount)
	for i := range ind.IndicesData {
		ind.IndicesData[i] = uint32(byteOrder.Uint32(buf[i*4 : (i+1)*4]))
	}
	ind.IndicesData = ind.decodeIndices(ind.IndicesData)
	return nil
}

func (ind *Indices32) Write(writer io.Writer) error {
	data := ind.encodeIndices(ind.IndicesData)
	buf := make([]byte, ind.CalcSize())
	offset := 0
	byteOrder.PutUint32(buf[offset:offset+4], uint32(ind.GetIndexCount()/3))
	offset += 4
	for i := range data {
		byteOrder.PutUint32(buf[offset:offset+4], data[i])
		offset += 4
	}
	if _, err := writer.Write(buf); err != nil {
		return nil
	}
	return nil
}

type EdgeIndices interface {
	Read(reader io.Reader) error
	Write(writer io.Writer) error
}

type EdgeIndices16 struct {
	EdgeIndices
	WestIndices  Indices16
	SouthIndices Indices16
	EastIndices  Indices16
	NorthIndices Indices16
}

func (ind *EdgeIndices16) Read(reader io.Reader) error {
	if err := ind.WestIndices.Read(reader); err != nil {
		return err
	}

	if err := ind.SouthIndices.Read(reader); err != nil {
		return err
	}

	if err := ind.EastIndices.Read(reader); err != nil {
		return err
	}

	if err := ind.NorthIndices.Read(reader); err != nil {
		return err
	}
	return nil
}

func (ind *EdgeIndices16) Write(writer io.Writer) error {
	if err := ind.WestIndices.Write(writer); err != nil {
		return err
	}

	if err := ind.SouthIndices.Write(writer); err != nil {
		return err
	}

	if err := ind.EastIndices.Write(writer); err != nil {
		return err
	}

	if err := ind.NorthIndices.Write(writer); err != nil {
		return err
	}
	return nil
}

type EdgeIndices32 struct {
	EdgeIndices
	WestIndices  Indices32
	SouthIndices Indices32
	EastIndices  Indices32
	NorthIndices Indices32
}

func (ind *EdgeIndices32) Read(reader io.Reader) error {
	if err := ind.WestIndices.Read(reader); err != nil {
		return err
	}

	if err := ind.SouthIndices.Read(reader); err != nil {
		return err
	}

	if err := ind.EastIndices.Read(reader); err != nil {
		return err
	}

	if err := ind.NorthIndices.Read(reader); err != nil {
		return err
	}
	return nil
}

func (ind *EdgeIndices32) Write(writer io.Writer) error {
	if err := ind.WestIndices.Write(writer); err != nil {
		return err
	}

	if err := ind.SouthIndices.Write(writer); err != nil {
		return err
	}

	if err := ind.EastIndices.Write(writer); err != nil {
		return err
	}

	if err := ind.NorthIndices.Write(writer); err != nil {
		return err
	}
	return nil
}

type QuantizedMeshTile struct {
	Header QuantizedMeshHeader
	Data   VertexData
	Index  interface{}
	Edge   interface{}
}

type MeshData struct {
	BBox     [2][3]float64
	Vertices [][3]float64
	Faces    [][3]int
}

func distance(max, min []float64) float64 {
	x := max[0] - min[0]
	y := max[1] - min[1]
	return math.Sqrt(x*x + y*y)
}

func (t *QuantizedMeshTile) setHeader(bbox [2][3]float64) {
	var c [3]float64
	c[0] = (bbox[1][0] + bbox[0][0]) / 2
	c[1] = (bbox[1][1] + bbox[0][1]) / 2
	c[2] = (bbox[1][2] + bbox[0][2]) / 2

	t.Header.CenterX = c[0]
	t.Header.CenterY = c[1]
	t.Header.CenterZ = c[2]

	t.Header.MinimumHeight = float32(bbox[0][2])
	t.Header.MaximumHeight = float32(bbox[1][2])

	t.Header.BoundingSphereCenterX = c[0]
	t.Header.BoundingSphereCenterY = c[1]
	t.Header.BoundingSphereCenterZ = c[2]
	t.Header.BoundingSphereRadius = distance(bbox[1][0:2], bbox[0][0:2])

	t.Header.HorizonOcclusionPointX = c[0]
	t.Header.HorizonOcclusionPointY = c[1]
	t.Header.HorizonOcclusionPointZ = bbox[1][2]
}

func (t *QuantizedMeshTile) getBBoxFromHeader() [2][3]float64 {
	tl :=
		[2]float64{t.Header.BoundingSphereCenterX - t.Header.BoundingSphereRadius,
			t.Header.BoundingSphereCenterY - t.Header.BoundingSphereRadius}
	br := [2]float64{t.Header.BoundingSphereCenterX + t.Header.BoundingSphereRadius,
		t.Header.BoundingSphereCenterY + t.Header.BoundingSphereRadius}

	return [2][3]float64{
		[3]float64{tl[0], tl[1], float64(t.Header.MinimumHeight)},
		[3]float64{br[0], br[1], float64(t.Header.MaximumHeight)},
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
		return nil, errors.New("mesh face error!")
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
		return nil, errors.New("mesh error!")
	}
	bbox := t.getBBoxFromHeader()
	vertices := t.getVertices(bbox)
	if faces, err := t.getFaces(); err != nil {
		return nil, err
	} else {
		return &MeshData{BBox: bbox, Vertices: vertices, Faces: faces}, nil
	}
}

func (t *QuantizedMeshTile) SetMesh(mesh MeshData, rescaled bool) {
	t.setHeader(mesh.BBox)

	var northlings []uint32
	var eastlings []uint32
	var southlings []uint32
	var westlings []uint32

	var us []uint16
	var vs []uint16
	var hs []uint16

	u := 0
	v := 0
	h := 0
	prevu := 0
	prevv := 0
	prevh := 0

	setflgs := make([]bool, len(mesh.Vertices))

	for i := range setflgs {
		setflgs[i] = false
	}

	for f := range mesh.Faces {
		for t := range mesh.Faces[f] {
			i := mesh.Faces[f][t]
			if setflgs[i] {
				continue
			}

			setflgs[i] = true

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
				westlings = append(westlings, uint32(i))
			} else if u == QUANTIZED_COORDINATE_SIZE {
				eastlings = append(eastlings, uint32(i))
			}

			if v == 0 {
				northlings = append(northlings, uint32(i))
			} else if v == QUANTIZED_COORDINATE_SIZE {
				southlings = append(southlings, uint32(i))
			}

			us = append(us, encodeZigZag(u-prevu))
			vs = append(vs, encodeZigZag(v-prevv))
			hs = append(hs, encodeZigZag(h-prevh))

			prevu = u
			prevv = v
			prevh = h
		}
	}

	t.Data.VertexCount = uint32(len(mesh.Vertices))
	t.Data.U = us
	t.Data.V = vs
	t.Data.H = hs

	if t.Data.VertexCount > 65535 {
		inds := make([]uint32, len(mesh.Faces)*3)
		for i := range mesh.Faces {
			inds[i*3] = uint32(mesh.Faces[i][0])
			inds[i*3+1] = uint32(mesh.Faces[i][1])
			inds[i*3+2] = uint32(mesh.Faces[i][2])
		}
		t.Index = Indices32{IndicesData: inds}
		t.Edge = EdgeIndices32{
			WestIndices:  Indices32{IndicesData: westlings},
			SouthIndices: Indices32{IndicesData: southlings},
			EastIndices:  Indices32{IndicesData: eastlings},
			NorthIndices: Indices32{IndicesData: northlings},
		}
	} else {
		inds := make([]uint16, len(mesh.Faces)*3)
		for i := range mesh.Faces {
			inds[i*3] = uint16(mesh.Faces[i][0])
			inds[i*3+1] = uint16(mesh.Faces[i][1])
			inds[i*3+2] = uint16(mesh.Faces[i][2])
		}
		westlings16 := make([]uint16, len(westlings))
		for i := range westlings {
			westlings16[i] = uint16(westlings[i])
		}
		southlings16 := make([]uint16, len(southlings))
		for i := range westlings {
			southlings16[i] = uint16(southlings[i])
		}
		eastlings16 := make([]uint16, len(eastlings))
		for i := range westlings {
			eastlings16[i] = uint16(eastlings[i])
		}
		northlings16 := make([]uint16, len(northlings))
		for i := range westlings {
			northlings16[i] = uint16(northlings[i])
		}

		t.Index = Indices16{IndicesData: inds}
		t.Edge = EdgeIndices16{
			WestIndices:  Indices16{IndicesData: westlings16},
			SouthIndices: Indices16{IndicesData: southlings16},
			EastIndices:  Indices16{IndicesData: eastlings16},
			NorthIndices: Indices16{IndicesData: northlings16},
		}
	}
}

func (t *QuantizedMeshTile) Read(reader io.Reader) error {
	if err := t.Header.Read(reader); err != nil {
		return err
	}
	if err := t.Data.Read(reader); err != nil {
		return err
	}
	if t.Data.VertexCount > 65535 {
		t.Index = Indices32{}
		t.Edge = EdgeIndices32{}
	} else {
		t.Index = Indices16{}
		t.Edge = EdgeIndices16{}
	}
	if err := t.Index.(Indices).Read(reader); err != nil {
		return err
	}
	if err := t.Edge.(EdgeIndices).Read(reader); err != nil {
		return err
	}
	return nil
}

func (t *QuantizedMeshTile) Write(writer io.Writer) error {
	var err error
	var offset int
	var alignment int
	if err = t.Header.Write(writer); err != nil {
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
			return nil
		}
	}
	if err = t.Index.(Indices).Write(writer); err != nil {
		return err
	}
	if err = t.Edge.(EdgeIndices).Write(writer); err != nil {
		return err
	}
	return nil
}
