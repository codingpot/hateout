import fs from 'fs/promises'
import { CharacterMap } from './characterMap'

describe('CharacterMap', function () {
  it('should be able to add a character', function () {
    const c = new CharacterMap()
    expect(c.adaptChar('a')).toBe(1)
  })

  it('should be able to look up a character', function () {
    const c = new CharacterMap()
    c.adaptChar('a')
    expect(c.lookupChar('a')).toBe(1)
  })

  it('should return undefined if the lookup fails', function () {
    const c = new CharacterMap()
    expect(c.lookupChar('a')).toBeUndefined()
  })

  it('should return a correct char for a correct ID', function () {
    const c = new CharacterMap()
    c.adaptChar('a')
    expect(c.lookupId(1)).toBe('a')
    expect(c.lookupId(0)).toBe('<PAD>')
  })

  describe('persistence', function () {
    const filename: string = 'test.json'
    afterEach(async () => {
      await fs.rm(filename, { force: true })
    })

    it('should save to a file in JSON', async function () {
      {
        const c = new CharacterMap()
        c.adaptChar('a')
        c.adaptChar('b')
        c.adaptChar('c')
        await c.saveToJSON('test.json')
      }
      {
        const c2 = await CharacterMap.loadFromJSON('test.json')
        expect(c2.lookupChar('<PAD>')).toBe(0)
        expect(c2.lookupChar('a')).toBe(1)
        expect(c2.lookupChar('b')).toBe(2)
        expect(c2.lookupChar('c')).toBe(3)

        // next ID should have been updated correctly.
        expect(c2.adaptChar('d')).toBe(4)
      }
    })
  })
})
