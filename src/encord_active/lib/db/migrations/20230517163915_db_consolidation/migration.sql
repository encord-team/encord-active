/*
  Warnings:

  - Added the required column `label_row_json` to the `LabelRow` table without a default value. This is not possible if the table is not empty.

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
    "location" TEXT,
    "label_row_json" TEXT NOT NULL
);
INSERT INTO "new_LabelRow" ("created_at", "data_hash", "data_title", "data_type", "id", "label_hash", "last_edited_at", "location") SELECT "created_at", "data_hash", "data_title", "data_type", "id", "label_hash", "last_edited_at", "location" FROM "LabelRow";
DROP TABLE "LabelRow";
ALTER TABLE "new_LabelRow" RENAME TO "LabelRow";
CREATE UNIQUE INDEX "LabelRow_label_hash_key" ON "LabelRow"("label_hash");
CREATE UNIQUE INDEX "LabelRow_data_hash_key" ON "LabelRow"("data_hash");
CREATE TABLE "new_DataUnit" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "data_hash" TEXT NOT NULL,
    "data_title" TEXT NOT NULL,
    "frame" INTEGER NOT NULL,
    "location" TEXT,
    "data_link" TEXT,
    "lr_data_hash" TEXT NOT NULL,
    CONSTRAINT "DataUnit_lr_data_hash_fkey" FOREIGN KEY ("lr_data_hash") REFERENCES "LabelRow" ("data_hash") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_DataUnit" ("data_hash", "data_title", "frame", "id", "location", "lr_data_hash") SELECT "data_hash", "data_title", "frame", "id", "location", "lr_data_hash" FROM "DataUnit";
DROP TABLE "DataUnit";
ALTER TABLE "new_DataUnit" RENAME TO "DataUnit";
CREATE UNIQUE INDEX "DataUnit_data_hash_frame_key" ON "DataUnit"("data_hash", "frame");
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
