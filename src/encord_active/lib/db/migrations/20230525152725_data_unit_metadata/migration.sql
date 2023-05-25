-- RedefineTables
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_DataUnit" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "data_hash" TEXT NOT NULL,
    "data_title" TEXT NOT NULL,
    "frame" INTEGER NOT NULL,
    "data_uri" TEXT,
    "lr_data_hash" TEXT NOT NULL,
    "width" INTEGER NOT NULL DEFAULT -1,
    "height" INTEGER NOT NULL DEFAULT -1,
    "fps" REAL NOT NULL DEFAULT -1,
    CONSTRAINT "DataUnit_lr_data_hash_fkey" FOREIGN KEY ("lr_data_hash") REFERENCES "LabelRow" ("data_hash") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_DataUnit" ("data_hash", "data_title", "data_uri", "frame", "id", "lr_data_hash") SELECT "data_hash", "data_title", "data_uri", "frame", "id", "lr_data_hash" FROM "DataUnit";
DROP TABLE "DataUnit";
ALTER TABLE "new_DataUnit" RENAME TO "DataUnit";
CREATE UNIQUE INDEX "DataUnit_data_hash_frame_key" ON "DataUnit"("data_hash", "frame");
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
