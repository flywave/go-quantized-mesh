// Copyright (c) 2017-present FlyWave, Inc. All Rights Reserved.
// See License.txt for license information.

package terrain

import (
	"os"
	"testing"
)

func TestLoader(t *testing.T) {
	f, err := os.Open("./testdata/5173.terrain")
	defer f.Close()

	if err != nil {
		t.Error("error")
	}

	tm := new(QuantizedMeshTile)
	err = tm.Read(f)

	if err != nil {
		t.Error("error")
	}

	if tm.Header.CenterX == 0 {
		t.Error("error")
	}
}
