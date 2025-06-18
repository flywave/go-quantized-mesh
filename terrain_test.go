package terrain

import (
	"fmt"
	"os"
	"testing"
)

func TestLoader(t *testing.T) {
	f, err := os.Open("./testdata/5173.terrain")

	if err != nil {
		t.Error("error")
	}
	defer f.Close()

	tm := new(QuantizedMeshTile)
	err = tm.Read(f, Ext_None)

	if err != nil {
		t.Error("error")
	}

	if tm.Header.CenterX == 0 {
		t.Error("error")
	}
}

func TestMime(t *testing.T) {

	f := GetTerrainMime(Ext_Light | Ext_WaterMask)

	if f == "" {
		t.FailNow()
	}
}

func TestOtc(t *testing.T) {
	n := 307
	n1 := decodeZigZag(uint16(n))
	fmt.Println(n1)
}
