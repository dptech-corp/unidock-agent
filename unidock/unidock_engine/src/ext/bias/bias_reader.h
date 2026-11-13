//
// Reader of the ud2_bias table (plan step 5c): the position-bias rows of one ligand, registered with the DMS
// reader's table registry (format/dms_reader.h) so read_ligand_dms_full and read_multi_ligand_dms_full read
// them after the core tables.
//

#ifndef UD2_EXT_BIAS_READER_H
#define UD2_EXT_BIAS_READER_H

#include "format/dms_reader.h"

/// The ud2_bias rows of the ligand `scope` describes, appended to out.bias.rows in rowid order
/// (SELECT atom_id, ref_x, ref_y, ref_z, v_set, r FROM ud2_bias [WHERE atom_id in the ligand's range] ORDER BY
/// rowid; a merged file's atom ids are rebased by scope.off). Registered for the table "ud2_bias".
void read_bias_rows(sqlite3* db, const TableScope& scope, UDFlexMol& out);

#endif // UD2_EXT_BIAS_READER_H
