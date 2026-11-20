// Copyright IBM Corp. 2015, 2025
// SPDX-License-Identifier: MPL-2.0

package iradix

import (
	"bytes"
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
)

// testKey encodes an id into a key, and testVal derives the value stored
// under that key. Readers use this to detect torn or foreign reads: any
// pair where the value does not match the key is dirty data.
func testKey(id int) []byte {
	return []byte(fmt.Sprintf("user/%06d", id))
}

func testVal(id int) int {
	return id * 31
}

func checkKeyVal(t *testing.T, k []byte, v int) bool {
	if !bytes.HasPrefix(k, []byte("user/")) {
		t.Errorf("foreign key read: %q", k)
		return false
	}
	id, err := strconv.Atoi(string(k[len("user/"):]))
	if err != nil {
		t.Errorf("malformed key read: %q", k)
		return false
	}
	if v != testVal(id) {
		t.Errorf("dirty read: key %q has value %d, want %d", k, v, testVal(id))
		return false
	}
	return true
}

// TestConcurrentTraversal hammers the tree with a single writer committing
// new snapshots while many readers traverse older snapshots concurrently.
// Readers must never observe torn, duplicated, unsorted, or foreign data:
// every node reachable from a snapshot must stay untouched for as long as
// any iterator references it.
func TestConcurrentTraversal(t *testing.T) {
	const (
		keyspace    = 4096
		readers     = 8
		writerRound = 3000
	)

	// Seed the initial tree.
	tree := New[int]()
	txn := tree.Txn()
	for i := 0; i < keyspace/2; i++ {
		txn.Insert(testKey(i), testVal(i))
	}
	tree = txn.Commit()

	var current atomic.Pointer[Tree[int]]
	current.Store(tree)

	// done signals readers to stop once the writer is finished.
	var done atomic.Bool
	var badReads atomic.Int64

	report := func() {
		badReads.Add(1)
	}

	var wg sync.WaitGroup

	// Single writer: churn the tree with inserts and deletes. Values are
	// always derived from the key so readers can validate what they read.
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := rand.New(rand.NewSource(1))
		for round := 0; round < writerRound; round++ {
			txn := current.Load().Txn()
			for op := 0; op < 8; op++ {
				id := rng.Intn(keyspace)
				if rng.Intn(2) == 0 {
					txn.Insert(testKey(id), testVal(id))
				} else {
					txn.Delete(testKey(id))
				}
			}
			current.Store(txn.Commit())
		}
		done.Store(true)
	}()

	// Readers: full forward iteration, reverse iteration, and prefix walks
	// over whatever snapshot is current. Every pass must be internally
	// consistent: strictly ordered, no duplicates, values matching keys.
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			for !done.Load() {
				snap := current.Load()

				switch r % 3 {
				case 0:
					it := snap.Root().Iterator()
					var prev []byte
					for k, v, ok := it.Next(); ok; k, v, ok = it.Next() {
						if !checkKeyVal(t, k, v) {
							report()
						}
						if prev != nil && bytes.Compare(prev, k) >= 0 {
							t.Errorf("forward iteration out of order: %q then %q", prev, k)
							report()
						}
						prev = append(prev[:0], k...)
					}
				case 1:
					it := snap.Root().ReverseIterator()
					var prev []byte
					for k, v, ok := it.Previous(); ok; k, v, ok = it.Previous() {
						if !checkKeyVal(t, k, v) {
							report()
						}
						if prev != nil && bytes.Compare(prev, k) <= 0 {
							t.Errorf("reverse iteration out of order: %q then %q", prev, k)
							report()
						}
						prev = append(prev[:0], k...)
					}
				case 2:
					var prev []byte
					snap.Root().WalkPrefix([]byte("user/00"), func(k []byte, v int) bool {
						if !checkKeyVal(t, k, v) {
							report()
						}
						if prev != nil && bytes.Compare(prev, k) >= 0 {
							t.Errorf("prefix walk out of order: %q then %q", prev, k)
							report()
						}
						prev = append(prev[:0], k...)
						return false
					})
				}
			}
		}(r)
	}

	wg.Wait()

	if badReads.Load() > 0 {
		t.Fatalf("observed %d dirty reads during concurrent traversal", badReads.Load())
	}

	// The writer is single-threaded, so the final snapshot must exactly
	// reflect every committed operation. Verify the final tree is coherent.
	final := current.Load()
	it := final.Root().Iterator()
	count := 0
	var prev []byte
	for k, v, ok := it.Next(); ok; k, v, ok = it.Next() {
		if !checkKeyVal(t, k, v) {
			t.Fatalf("final tree contains dirty data")
		}
		if prev != nil && bytes.Compare(prev, k) >= 0 {
			t.Fatalf("final tree iteration out of order")
		}
		prev = append(prev[:0], k...)
		count++
	}
	if count != final.Len() {
		t.Fatalf("final tree iterated %d keys, Len() = %d", count, final.Len())
	}
}
