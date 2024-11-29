// eslint-disable-next-line no-unused-vars
import React, { useState } from "react";
import "./App.css";
import OpenAI from "openai";

const pokemonTypes = [
  "Fire", "Water", "Grass", "Electric", "Bug", "Normal", "Psychic", "Fairy",
  "Ghost", "Dark", "Dragon", "Ice", "Fighting", "Steel", "Poison", "Flying",
];

const apiKey = process.env.REACT_API_KEY;

const App = () => {
  const [type1, setType1] = useState("Fire");
  const [type2, setType2] = useState("");
  const [pokemonData, setPokemonData] = useState();
  const [loading, setLoading] = useState(false);

  const generatePokemonInfo = async () => {
    setLoading(true); // Show spinner
    setPokemonData(null); // Clear previous result

    try {
      const openai = new OpenAI({
        dangerouslyAllowBrowser: true,
        apiKey: apiKey, // Replace with your API key
        baseURL: "https://api.pawan.krd/v1", // Hosted API base URL
      });

      const chatCompletion = await openai.chat.completions.create({
        messages: [
          {
            role: "user",
            content: `
            Generate a clean JSON object for a Pokémon with the following structure:
            {
              "name": "string",
              "abilities": ["string", "string"],
              "description": "string",
              "HP": int,
              "Attack": int,
              "SpecialAttack": int,
              "Defense": int
              }
              Respond with only the JSON object, without comments, single quotes, or any extra text.,
              `
          },
        ],
        model: "pai-001", // Ensure this model is supported by the API
      });

      const responseText = chatCompletion.choices[0]?.message?.content?.trim();
      console.log("Raw response:", responseText);

      // Step 1: Extract only the JSON part
      const jsonMatch = responseText.match(/{.*}/s); // Match the JSON object
      if (!jsonMatch) {
        throw new Error("No valid JSON object found in the response.");
      }

      let cleanedJSON = jsonMatch[0]
        .replace(/\/\/.*$/gm, "") // Remove single-line comments
        .replace(/\/\*[\s\S]*?\*\//g, "") // Remove block comments
        .replace(/,\s*}/g, "}") // Remove trailing commas
        .replace(/'/g, '"'); // Replace all single quotes with double quotes

      console.log("Cleaned JSON:", cleanedJSON);

      // Step 2: Parse the cleaned JSON
      const parsedData = JSON.parse(cleanedJSON);
      setPokemonData(parsedData); // Update state with cleaned JSON

      console.log("Parsed Pokémon card:", parsedData);
    } catch (error) {
      console.error("Error generating Pokémon card:", error);
    } finally {
      setLoading(false); // Hide spinner
    }
  };

  const clearPokemon = () => {
    setPokemonData(null);
  };

  const getImagePath = (type) => {
    return `/src/assets/templates/${type}.png`;
  };

  return (
    <div className="app">
      <div className="layout">
        {/* Type Selector on the Left */}
        <div className="type-selector">
          <label>
            Primary Type:
            <select value={type1} onChange={(e) => setType1(e.target.value)}>
              {pokemonTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>

          <label>
            Secondary Type:
            <select value={type2} onChange={(e) => setType2(e.target.value)}>
              <option value="">None</option>
              {pokemonTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>
          <button onClick={generatePokemonInfo}>Generate Pokémon</button>
          {pokemonData && <button onClick={clearPokemon}>Generate New Pokémon</button>}
        </div>

        {/* Card Display */}
        <div className="card-container">
          {loading && <div className="spinner">Loading...</div>}

          {pokemonData && (
            <div className="card">
              <img
                src={getImagePath(type1)}
                alt={type1}
                className="background-image"
              />

              {/* GAN Image Container */}
              {pokemonData && (
                <div className="gan-image-container">
                  <img
                    src="/assets/templates/Fire.png" // Replace with the actual GAN image source
                    alt="GAN Generated"
                    className="gan-image"
                  />
                </div>
              )}

              <h2>{pokemonData.name}</h2>
              <p>{pokemonData.description}</p>
              <div className="stats-text">

                <p>
                  Name: {pokemonData.name}
                  <br></br>
                  Type: {type1}
                  {type2 ? ` / ${type2}` : ""}
                </p>

                <div className="stats">
                  <p>HP: {pokemonData.HP}</p>
                  <p>Attack: {pokemonData.Attack}</p>
                  <p>Special Attack: {pokemonData.SpecialAttack}</p>
                  <p>Defense: {pokemonData.Defense}</p>
                </div>

                <div className="abilities">
                  <h3>Abilities:</h3>
                  {pokemonData.abilities.map((ability, index) => (
                    <p key={index}>{ability}</p>
                  ))}
                </div>
              </div>


            </div>

          )}
        </div>
      </div>
    </div>
  );
};

export default App;
