module.exports = {
  siteMetadata: {
    title: `Karter's re:Trak`,
    name: `re:Trak`,
    siteUrl: `https://karter.io/`,
    description: `Karter's latest news on Natural Language Processing`,
    hero: {
      heading: `For the latest news on Natural Language Processing`,
      maxWidth: 652,
    },
    social: [
      {
        name: `facebook`,
        url: `https://www.facebook.com/monthly.nlp`,
      },
      {
        name: `github`,
        url: `https://github.com/huffon`,
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
    {
      resolve: `gatsby-plugin-disqus`,
      options: {
        shortname: `https-re-trak-github-io`
      }
    },
    {
      resolve: `gatsby-plugin-gtag`,
      options: {
        trackingId: `G-1ZMN9450YB`, // 측정 ID
        head: true,
        anonymize: true,
      },
    },
    {
      resolve: `gatsby-plugin-google-gtag`,
      options: {
        // You can add multiple tracking ids and a pageview event will be fired for all of them.
        trackingIds: [
          "G-1ZMN9450YB", // Google Analytics / GA
          // "AW-CONVERSION_ID", // Google Ads / Adwords / AW
        ],
      },
    }
  ],
};
