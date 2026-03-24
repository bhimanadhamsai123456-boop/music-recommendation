const form = document.getElementById("recommend-form");
const input = document.getElementById("song_name");
const message = document.getElementById("form-message");
const results = document.getElementById("results");
const badge = document.getElementById("result-badge");

function renderPlaceholder(text) {
  results.innerHTML = `
    <article class="placeholder-card">
      <p class="placeholder-title">${text}</p>
      <p class="placeholder-text">Try another search or connect your recommender endpoint.</p>
    </article>
  `;
}

function renderTracks(query, items, source) {
  if (!items.length) {
    renderPlaceholder(`No recommendations found for "${query}".`);
    badge.textContent = "No matches";
    return;
  }

  results.innerHTML = items
    .map(
      (item, index) => `
        <article class="track-card">
          <span class="track-index">${index + 1}</span>
          ${item.artwork_url ? `<img class="track-art" src="${item.artwork_url}" alt="${item.title} artwork">` : ""}
          <h4>${item.title}</h4>
          <p class="track-meta">${item.artist}</p>
          <p class="track-meta">${item.album}</p>
          <p class="track-meta">${item.year ? `Year: ${item.year}` : ""}${item.popularity ? ` • Popularity: ${item.popularity}` : ""}</p>
          <p class="track-meta">${item.language ? `Language: ${item.language}` : ""}</p>
          ${item.track_url ? `<a class="track-link" href="${item.track_url}" target="_blank" rel="noreferrer">Open in Spotify</a>` : ""}
        </article>
      `
    )
    .join("");

  badge.textContent = `${items.length} tracks • ${source}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const songName = input.value.trim();

  if (!songName) {
    message.textContent = "Please enter a song name first.";
    return;
  }

  message.textContent = "Generating recommendations...";
  badge.textContent = "Loading";

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ song_name: songName }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Unable to fetch recommendations.");
    }

    renderTracks(data.query, data.recommendations || [], data.source || "api");
    message.textContent = `Showing recommendations for "${data.query}".`;
  } catch (error) {
    renderPlaceholder("The recommender service is not responding.");
    badge.textContent = "Error";
    message.textContent = error.message;
  }
});
