package terrain

import (
	"encoding/json"

	vec3d "github.com/flywave/go3d/float64/vec3"
)

var (
	EXT_LIGHT_HEADER        = ExtensionHeader{ExtensionId: QUANTIZED_MESH_LIGHT_EXTENSION_ID}
	EXT_WATERMASK_HEADER    = ExtensionHeader{ExtensionId: QUANTIZED_MESH_WATERMASK_EXTENSION_ID}
	EXT_METADATA_HEADER     = ExtensionHeader{ExtensionId: QUANTIZED_MESH_METADATA_EXTENSION_ID}
	EXT_FACEGROUP_HEADER    = ExtensionHeader{ExtensionId: QUANTIZED_MESH_FACEGROUP_EXTENSION_ID}
	EXT_DISCARDFACES_HEADER = ExtensionHeader{ExtensionId: QUANTIZED_MESH_DISCARDFACES_EXTENSION_ID}
)

type ExtensionHeader struct {
	ExtensionId     uint8
	ExtensionLength uint32
}

type OctEncodedVertexNormals struct {
	Norm []vec3d.T
}

type WaterMaskLand struct {
	Mask uint8
}

type WaterMask struct {
	Mask [256 * 256]uint8
}

type Metadata struct {
	JsonLength uint32
	Json       json.RawMessage
}

type FaceGroop struct {
	Id    int
	Start uint32
	End   uint32
}
