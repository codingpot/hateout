import fs from 'fs/promises'

export class CharacterMap {
  private map: Map<string, number> = new Map()
  private reverseMap: Map<number, string> = new Map()
  private nextId: number = 0

  constructor() {
    const id = this.nextId++
    this.map.set('<PAD>', id)
    this.reverseMap.set(id, '<PAD>')
  }

  adaptChar(char: string): number {
    if (this.map.has(char)) {
      return this.map.get(char)!
    }
    const id = this.nextId++
    this.map.set(char, id)
    this.reverseMap.set(id, char)
    return id
  }

  lookupChar(char: string): number | undefined {
    return this.map.get(char)
  }

  lookupId(id: number): string | undefined {
    return this.reverseMap.get(id)
  }

  saveToJSON(file: string): Promise<void> {
    const json = {
      map: Array.from(this.map.entries()),
      reverseMap: Array.from(this.reverseMap.entries()),
    }
    return fs.writeFile(file, JSON.stringify(json), 'utf-8')
  }

  static async loadFromJSON(file: string): Promise<CharacterMap> {
    const json = JSON.parse(await fs.readFile(file, 'utf-8'))

    const c = new CharacterMap()
    c.map = new Map(json.map)
    c.reverseMap = new Map(json.reverseMap)
    c.nextId = Math.max(...Array.from(c.map.values())) + 1
    return c
  }
}
