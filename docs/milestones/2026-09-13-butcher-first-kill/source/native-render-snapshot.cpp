// A display-only companion. The exported function may ONLY run in a disposable
// fork child. All graphical loading and state changes stay in that child.
// Reuse the pinned engine's actual dungeon renderer, not a reconstructed scene.
#include "engine/render/scrollrt.cpp"
#include "engine/palette.h"
#include "engine/load_cel.hpp"
#include "objects.h"
#include "player.h"
#include "items.h"
#include "monster.h"
#include <cstdio>
#include <stdexcept>

extern "C" __attribute__((visibility("default")))
int alphadiablo_render_snapshot_v1(const char *filename)
{
    using namespace devilution;
    try {
        // Never call this function in the live parent or run a game tick here.
        HeadlessMode = false;
        GetOptions().Graphics.zoom.SetValue(false);
        GetOptions().Graphics.perPixelLighting.SetValue(false);
        gnScreenWidth = 640;
        gnScreenHeight = 480;
        gnViewportHeight = 352;
        ProgressToNextGameTick = 0;
        if (leveltype != DTYPE_CATHEDRAL)
            throw std::runtime_error("capture supports cathedral scenes only");
        LoadPaletteAndInitBlending("levels\\l1data\\l1_1.pal");
        auto monsters = InitAllMonsterGFX();
        if (!monsters) throw std::runtime_error(monsters.error());
        auto objects = InitObjectGFX();
        if (!objects) throw std::runtime_error(objects.error());
        auto missiles = InitMissileGFX();
        if (!missiles) throw std::runtime_error(missiles.error());
        SetUpMissileAnimationData();
        InitItemGFX();
        for (size_t i = 0; i < ActiveItemCount; ++i) {
            Item &item = Items[ActiveItems[i]];
            const Item before = item;
            item.setNewAnimation(false);
            const auto sprites = item.AnimInfo.sprites;
            item = before;
            item.AnimInfo.sprites = sprites;
        }
        for (size_t i = 0; i < ActiveObjectCount; ++i)
            SyncObjectAnim(Objects[ActiveObjects[i]]);
        InitPlayerGFX(*MyPlayer);
        SyncPlrAnim(*MyPlayer);
        // Do not rebuild the corpse table: by this point some unique monsters
        // have been removed from ActiveMonsters, and rebuilding renumbers their
        // corpse slots. Hydrate graphics in the original snapshot's slots.
        for (size_t i = 0; i < LevelMonsterTypeCount; ++i) {
            const CMonster &type = LevelMonsterTypes[i];
            if (type.corpseId > 0 && type.corpseId <= MaxCorpses)
                Corpses[type.corpseId - 1].sprites = type.getAnimData(MonsterGraphic::Death).sprites;
        }
        for (Corpse &corpse : Corpses) {
            if (corpse.translationPaletteIndex != 0) {
                // Monster slots can be reused after a town return. The corpse
                // retains its original animation frame count and width, not a
                // stable type pointer. Require an unambiguous matching native
                // death animation; never substitute an arbitrary monster.
                int matches = 0;
                for (size_t i = 0; i < LevelMonsterTypeCount; ++i) {
                    const auto &death = LevelMonsterTypes[i].getAnimData(MonsterGraphic::Death);
                    if (death.frames - 1 == corpse.frame && death.width == corpse.width) {
                        corpse.sprites = death.sprites;
                        ++matches;
                    }
                }
                if (matches != 1)
                    throw std::runtime_error("ambiguous unique-corpse display binding");
            }
        }
        if (stonendx > 0 && stonendx <= MaxCorpses)
            Corpses[stonendx - 1].sprites.emplace(*GetMissileSpriteData(MissileGraphicID::StoneCurseShatter).sprites);
        for (int x = 0; x < MAXDUNX; ++x) {
            for (int y = 0; y < MAXDUNY; ++y) {
                const int id = dCorpse[x][y] & 0x1F;
                if (id != 0 && !Corpses[id - 1].sprites)
                    throw std::runtime_error("unbound corpse graphics in snapshot");
                if (id != 0) {
                    const Corpse &corpse = Corpses[id - 1];
                    const auto sprites = corpse.spritesForDirection(static_cast<Direction>((dCorpse[x][y] >> 5) & 7));
                    if (corpse.frame < 0 || static_cast<unsigned>(corpse.frame) >= sprites.numSprites())
                        throw std::runtime_error("corpse frame outside loaded native animation");
                }
            }
        }
        pSpecialCels = LoadCel("levels\\l1data\\l1s", 64);
        CalcViewportGeometry();
        SDL_Surface *pixels = SDL_CreateRGBSurfaceWithFormat(0, 640, 352, 8, SDL_PIXELFORMAT_INDEX8);
        if (pixels == nullptr) throw std::runtime_error(SDL_GetError());
        SDL_SetPaletteColors(pixels->format->palette, logical_palette.data(), 0, 256);
        SDL_FillRect(pixels, nullptr, 0);
        Point first = MyPlayer->position.tile;
        Displacement offset {};
        CalcFirstTilePosition(first, offset);
        DrawGame(Surface(pixels), first, offset);
        const int result = SDL_SaveBMP(pixels, filename);
        SDL_FreeSurface(pixels);
        if (result != 0) throw std::runtime_error(SDL_GetError());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "snapshot render failed: %s\n", error.what());
        return 1;
    }
}
