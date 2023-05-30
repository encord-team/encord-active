/*
  Warnings:

  - You are about to drop the column `location` on the `LabelRow` table. All the data in the column will be lost.
  - You are about to drop the column `location` on the `DataUnit` table. All the data in the column will be lost.

*/
-- RedefineTables
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_LabelRow" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "label_hash" TEXT,
    "data_hash" TEXT NOT NULL,
    "data_title" TEXT NOT NULL,
    "data_type" TEXT NOT NULL,
    "created_at" TEXT NOT NULL,
    "last_edited_at" TEXT NOT NULL,
    "label_row_json" TEXT
);
INSERT INTO "new_LabelRow" ("created_at", "data_hash", "data_title", "data_type", "id", "label_hash", "last_edited_at") SELECT "created_at", "data_hash", "data_title", "data_type", "id", "label_hash", "last_edited_at" FROM "LabelRow";
DROP TABLE "LabelRow";
ALTER TABLE "new_LabelRow" RENAME TO "LabelRow";
CREATE UNIQUE INDEX "LabelRow_label_hash_key" ON "LabelRow"("label_hash");
CREATE UNIQUE INDEX "LabelRow_data_hash_key" ON "LabelRow"("data_hash");
CREATE TABLE "new_DataUnit" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "data_hash" TEXT NOT NULL,
    "data_title" TEXT NOT NULL,
    "frame" INTEGER NOT NULL,
    "data_uri" TEXT,
    "lr_data_hash" TEXT NOT NULL,
    CONSTRAINT "DataUnit_lr_data_hash_fkey" FOREIGN KEY ("lr_data_hash") REFERENCES "LabelRow" ("data_hash") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_DataUnit" ("data_hash", "data_title", "frame", "id", "lr_data_hash") SELECT "data_hash", "data_title", "frame", "id", "lr_data_hash" FROM "DataUnit";
DROP TABLE "DataUnit";
ALTER TABLE "new_DataUnit" RENAME TO "DataUnit";
CREATE UNIQUE INDEX "DataUnit_data_hash_frame_key" ON "DataUnit"("data_hash", "frame");
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
