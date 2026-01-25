---
layout: page
title: Blog
---

<div class="blog-toggle">
  <span id="toggle-date" class="toggle-option active" onclick="showDateView()">By date</span>
  <span class="toggle-separator">|</span>
  <span id="toggle-tag" class="toggle-option" onclick="showTagView()">By tag</span>
</div>

<!-- Date View (default) -->
<div id="date-view">
  <ul class="posts">
    {% for post in site.posts %}

      {% unless post.next %}
        <h3>{{ post.date | date: '%Y' }}</h3>
      {% else %}
        {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
        {% capture nyear %}{{ post.next.date | date: '%Y' }}{% endcapture %}
        {% if year != nyear %}
          <h3>{{ post.date | date: '%Y' }}</h3>
        {% endif %}
      {% endunless %}

      <li itemscope>
        <a href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
        <div class="post-date">{{ post.date | date: "%B %-d" }}</div>
      </li>

    {% endfor %}
  </ul>
</div>

<!-- Tag View -->
<div id="tag-view" style="display: none;">
  <div class="tag-cloud">
    {% assign sorted_tags = site.tags | sort %}
    {% for tag in sorted_tags %}
      <span class="tag-item" onclick="filterByTag('{{ tag[0] | escape }}')">
        {{ tag[0] }} ({{ tag[1].size }})
      </span>
    {% endfor %}
  </div>

  <div id="tag-posts" class="tag-posts">
    <p class="tag-instruction">Click a tag above to see related posts.</p>
  </div>

  <!-- Hidden data for JavaScript -->
  <script type="application/json" id="posts-data">
  [
    {% for post in site.posts %}
    {
      "title": {{ post.title | jsonify }},
      "url": "{{ site.github.url }}{{ post.url }}",
      "date": "{{ post.date | date: '%B %-d, %Y' }}",
      "tags": {{ post.tags | jsonify }}
    }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ]
  </script>
</div>

<script>
function showDateView() {
  document.getElementById('date-view').style.display = 'block';
  document.getElementById('tag-view').style.display = 'none';
  document.getElementById('toggle-date').classList.add('active');
  document.getElementById('toggle-tag').classList.remove('active');
}

function showTagView() {
  document.getElementById('date-view').style.display = 'none';
  document.getElementById('tag-view').style.display = 'block';
  document.getElementById('toggle-date').classList.remove('active');
  document.getElementById('toggle-tag').classList.add('active');
}

function filterByTag(selectedTag) {
  const postsData = JSON.parse(document.getElementById('posts-data').textContent);
  const tagPostsDiv = document.getElementById('tag-posts');

  // Clear previous selection styling
  document.querySelectorAll('.tag-item').forEach(item => {
    item.classList.remove('tag-selected');
  });

  // Highlight selected tag
  document.querySelectorAll('.tag-item').forEach(item => {
    if (item.textContent.trim().startsWith(selectedTag + ' ')) {
      item.classList.add('tag-selected');
    }
  });

  // Filter posts by tag
  const filteredPosts = postsData.filter(post =>
    post.tags && post.tags.includes(selectedTag)
  );

  // Build HTML
  let html = '<h3>' + selectedTag + '</h3><ul class="posts">';
  filteredPosts.forEach(post => {
    html += '<li><a href="' + post.url + '">' + post.title + '</a>';
    html += '<div class="post-date">' + post.date + '</div></li>';
  });
  html += '</ul>';

  tagPostsDiv.innerHTML = html;
}
</script>
