//
// Reader of the aromatic-ring bias tables ud2_ring_bias + ud2_ring_bias_atoms (plan step 6): the ring rows of one
// ligand, the second entry of the DMS reader's table registry (format/dms_reader.h), with the whole-file checks of
// a merged read. The October reader of format/dms_reader.cpp (read_ring_bias, count_merged_ring_bias_rows and the
// orphan check of read_multi_ligand_dms_full), same SQL, same column reads, same row order and errors.
//

#ifndef UD2_EXT_RING_READER_H
#define UD2_EXT_RING_READER_H

#include "format/dms_reader.h"

/// The ring rows of the ligand `scope` describes, into out.ring.rows (cleared first): rows in ring_row order, each
/// ring's atoms in pos order. A single-ligand file is read whole; in a merged file a ring belongs to the ligand
/// owning its atoms. Atom ids in [off, off + natom) are rebased to 0 and the template normal is normalised. Throws on
/// one table without the other, a ring that is not 5- or 6-membered, a ring atom outside the ligand, or a zero
/// template normal. Registered for "ud2_ring_bias" with the companion "ud2_ring_bias_atoms".
void read_ring_bias_rows(sqlite3* db, const TableScope& scope, UDFlexMol& out);

/// Rows of ud2_ring_bias in a merged DMS (0 without the table). ring_row joins the two ring tables across every
/// ligand of the file, so a repeated ring_row (per-ligand ud2align files each number theirs from 0) would pair one
/// ligand's ring with another's atoms: throws.
int count_merged_ring_bias_rows(sqlite3* db);

#endif // UD2_EXT_RING_READER_H
