module.exports = {
  siteMetadata: {
    title: `re-Trak`,
    name: `re-Track`,
    siteUrl: `https://re-trak.github.io/`,
    description: `This is my description that will be used in the meta tags and important for search results`,
    hero: {
      heading: `For the latest news on Natural Language Processing`,
      maxWidth: 652,
    },
    social: [
      {
        name: `twitter`,
        url: `https://twitter.com/huffonism`,
      },
      {
        name: `github`,
        url: `https://github.com/huffon`,
      },
      {
        name: `instagram`,
        url: `https://instagram.com/huffonism`,
      },
      {
        name: `linkedin`,
        url: `https://www.linkedin.com/in/huffonism`,
      },
    ],
  },
  plugins: [
    {
      resolve: "@narative/gatsby-theme-novela",
      options: {
        contentPosts: "content/posts",
        contentAuthors: "content/authors",
        basePath: "/",
        authorsPage: true,
        mailchimp: true,
        sources: {
          local: true,
          // contentful: true,
        },
      },
    },
    {
      resolve: 'gatsby-plugin-mailchimp',
      options: {
        endpoint: 'https://github.us1.list-manage.com/subscribe/post?u=f787805e9797187417c706c5b&amp;id=b11a0ea552', // add your MC list endpoint here; see plugin repo for instructions
      },
    },
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: `Novela by Narative`,
        short_name: `Novela`,
        start_url: `/`,
        background_color: `#fff`,
        theme_color: `#fff`,
        display: `standalone`,
        icon: `src/assets/favicon.png`,
      },
    },
    {
      resolve: `gatsby-plugin-netlify-cms`,
      options: {
      },
    },
  ],
};
