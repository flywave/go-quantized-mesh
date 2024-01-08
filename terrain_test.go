package terrain

import (
	"fmt"
	"os"
	"testing"
)

func TestLoader(t *testing.T) {
	f, err := os.Open("./testdata/11503.terrain")
	defer f.Close()

	if err != nil {
		t.Error("error")
	}

	tm := new(QuantizedMeshTile)
	err = tm.Read(f, Ext_Light_WaterMask)

	if err != nil {
		t.Error("error")
	}

	if tm.Header.CenterX == 0 {
		t.Error("error")
	}
}

func TestMime(t *testing.T) {

	f := GetTerrainMime(Ext_Light | Ext_WaterMask)

	if f != "" {
		t.FailNow()
	}
}

func TestOtc(t *testing.T) {
	n := 307
	n1 := decodeZigZag(uint16(n))
	fmt.Println(n1)
}
